//
// Created by Eduardo Montilva on 24/7/25.
//

#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "raylib.h"
#include <time.h>

#define NUM_DISKS 5
#define NUM_PEGS 3
#define MAX_MOVEMENTS 1000

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

// Log struct for PufferLib
typedef struct {
    float perf; // 0-1 normalized performance metric
    float score; // unnormalized score
    float episode_return; // sum of agent rewards over episode
    float episode_length; // number of steps in episode
    float n; // Required as the last field
    float episode_movements; // Number of movements in the episode
} Log;

typedef struct {
    int disks[NUM_DISKS]; // Array to hold the disks on each peg
    int disk_count; // Number of disks on the peg
} Peg;

// Environment struct
typedef struct {
    Log log; // Required field
    float* observations; // state of each peg
    int* actions; // [current]
    float* rewards; // [reward]
    unsigned char* terminals; // [done]
    unsigned char* truncations; // [truncated] (optional, but recommended)

    Peg pegs[NUM_PEGS];

    Color disk_colors[NUM_DISKS]; // Colors for each disk

    int moves;
    int size;
} Hanoi;

// Add episode statistics to log
void add_log(Hanoi* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_length += 1.0f;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
    env->log.episode_movements += env->moves;
}

float random_float(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

int random_int(int low, int high) {
    return low + rand() % (high - low + 1);
}

void init_disk_colors(Hanoi* env) {
    // Initialize colors for disks using random colors
    for(int i = 0; i < NUM_DISKS; i++) {
        env->disk_colors[i].r = (unsigned char)random_float(0, 255);
        env->disk_colors[i].g = (unsigned char)random_float(0, 255);
        env->disk_colors[i].b = (unsigned char)random_float(0, 255);
        env->disk_colors[i].a = 255; // Fully opaque
    }
}

void c_reset(Hanoi* env) {

    // Reset
    env->actions[0] = 0; // Always start at first peg
    env->actions[1] = random_int(1, NUM_PEGS - 1); // Randomly select to peg except first one

    // Clear pegs
    memset(env->pegs, 0, sizeof(Peg) * NUM_PEGS);
    for(int i = 0; i < NUM_PEGS; i++) {
        env->pegs[i].disk_count = 0;
        memset(env->pegs[i].disks, -1, sizeof(int) * NUM_DISKS); // Initialize with -1 (no disks)
    }

    // Place all disks in the first peg
    for(int i = 0; i < NUM_DISKS; i++) {
        int disk_idx = NUM_DISKS - i - 1;
        env->pegs[0].disks[i] = disk_idx; // From largest to smallest
        env->pegs[0].disk_count++;
    }

    // First, reset observations to an "empty" value like -1
    for (int i = 0; i < NUM_PEGS * NUM_DISKS; i++) {
        env->observations[i] = -1.0f;
    }

    // Then, fill in the current disk positions
    for (int i = 0; i < NUM_PEGS; i++) { // For each peg
        for (int j = 0; j < env->pegs[i].disk_count; j++) { // For each disk on that peg
            int disk_id = env->pegs[i].disks[j];
            // The observation encodes the peg, the position in the stack, and the disk id
            env->observations[i * NUM_DISKS + j] = (float)disk_id;
        }
    }

    env->moves = 0;
    env->rewards[0] = 0.0f; // Reset reward

    init_disk_colors(env);
}

void c_step(Hanoi* env) {

    env->moves++;

    // printf("STEP!\n");
    int from_peg = env->actions[0];
    int to_peg = env->actions[1];
    // printf("Checking if moving disk from peg %d to peg %d is possible\n", from_peg, to_peg);
    if(env->pegs[from_peg].disk_count == 0) {
        // No disks to move from the selected peg
        // printf("Attempting to move from an empty peg: %d\n", from_peg);
        env->rewards[0] -= 0.01f; // Invalid action

        env->terminals[0] = 0;
        if(env->truncations) {
            env->truncations[0] = 0;
        }

        return;
    }
    // printf("Possible: Current disk count on peg %d: %d\n", from_peg, env->pegs[from_peg].disk_count);

    // Get top disk
    int disk_to_move = env->pegs[from_peg].disks[env->pegs[from_peg].disk_count - 1];

    // If to_peg is not empty, check if move is valid (no larger disk on top)
    if(env->pegs[to_peg].disk_count > 0) {
        int top_disk = env->pegs[to_peg].disks[env->pegs[to_peg].disk_count - 1];
        if(disk_to_move > top_disk) {
            // Invalid move, larger disk on top
            env->rewards[0] = -1.0f; // Assign a smaller, but significant, penalty
            return;
        }
    }

    // Move disk
    // printf("Moving disk %d from peg %d to peg %d\n", disk_to_move, from_peg, to_peg);
    env->pegs[from_peg].disks[env->pegs[from_peg].disk_count - 1] = -1; // Remove disk from the peg
    env->pegs[from_peg].disk_count--;
    env->pegs[to_peg].disks[env->pegs[to_peg].disk_count] = disk_to_move;
    env->pegs[to_peg].disk_count++;
    // printf("Disk %d moved successfully.\n", disk_to_move);

    // Check if game is won
    bool terminated = env->pegs[NUM_PEGS - 1].disk_count == NUM_DISKS;
    // Check truncated
    bool truncated = env->moves >= MAX_MOVEMENTS;
    bool done = terminated || truncated;
    // printf("Game terminated: %d, truncated: %d\n", terminated, truncated);

    // We reward the agent by making less moves
    //env->rewards[0] = truncated ? -1.0f : 1.0f - ((float)env->moves / MAX_MOVEMENTS); // less movements, higher reward
    if (terminated) {
        env->rewards[0] = 10.0f*(1.0f - (float)env->moves / (float)MAX_MOVEMENTS); // Large reward for winning, but also penalize for moves
    } else if (truncated) {
        env->rewards[0] = -5.0f; // Larger penalty for running out of moves
    } else {
        // Give a small positive reward for making a valid move.
        env->rewards[0] = 0.1f - (0.01f * env->moves);
    }
    env->terminals[0] = terminated ? 1 : 0;
    if(env->truncations) {
        env->truncations[0] = truncated ? 1 : 0;
    }

    if(done) {
        add_log(env);
        c_reset(env);
    } else {
        // Update observations
        // printf("Updating observations...\n");

        // First, reset observations to an "empty" value
        for (int i = 0; i < NUM_PEGS * NUM_DISKS; i++) {
            env->observations[i] = -1.0f;
        }

        // Then, fill in the current disk positions
        for (int i = 0; i < NUM_PEGS; i++) { // For each peg
            for (int j = 0; j < env->pegs[i].disk_count; j++) { // For each disk on that peg
                int disk_id = env->pegs[i].disks[j];
                // The observation encodes the peg, the position in the stack, and the disk id
                env->observations[i * NUM_DISKS + j] = (float)disk_id;
            }
        }
        // printf("Observations updated.\n");
    }
}

void c_close(Hanoi* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void c_render(Hanoi* env) {
    if (!IsWindowReady()) {
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PufferLib Hanoi Tower");
        SetTargetFPS(30);
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw the base platform
    DrawRectangle(50, SCREEN_HEIGHT - 100, SCREEN_WIDTH - 100, 20, DARKGRAY);

    // --- Draw Pegs ---
    int peg_width = 20;
    int peg_height = 300;
    int peg_y = SCREEN_HEIGHT - 100 - peg_height;

    for (int i = 0; i < NUM_PEGS; i++) {
        int peg_x = (SCREEN_WIDTH / (NUM_PEGS + 1)) * (i + 1);
        DrawRectangle(peg_x - peg_width / 2, peg_y, peg_width, peg_height, GRAY);
    }

    // --- Draw Disks ---
    int disk_height = 25;
    for (int i = 0; i < NUM_PEGS; i++) {
        int peg_x = (SCREEN_WIDTH / (NUM_PEGS + 1)) * (i + 1);
        for (int j = 0; j < env->pegs[i].disk_count; j++) {
            int disk_size = env->pegs[i].disks[j];
            int disk_width = 30 + disk_size * 25;
            int disk_x = peg_x - disk_width / 2;
            int disk_y = SCREEN_HEIGHT - 100 - (j + 1) * disk_height;

            // Use a color based on the disk size
            Color disk_color = env->disk_colors[disk_size % 10];
            DrawRectangle(disk_x, disk_y, disk_width, disk_height, disk_color);
            DrawRectangleLines(disk_x, disk_y, disk_width, disk_height, DARKGRAY);
        }
    }

    char moves_text[32];
    snprintf(moves_text, sizeof(moves_text), "Moves = %d", env->moves);
    DrawText(moves_text, 10, 10, 20, (Color){255, 255, 255, 255});

    char reward_text[32];
    snprintf(reward_text, sizeof(moves_text), "Reward = %.4f", env->rewards[0]);
    DrawText(reward_text, 10, 40, 20, (Color){255, 255, 255, 255});


    EndDrawing();

    // Pause execution so we can see the movements
    struct timespec req;
    req.tv_sec = 0;
    req.tv_nsec = 100 * 1000 * 1000; // 500 ms
    nanosleep(&req, NULL);
}