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

#define MAX_MOVEMENTS 150

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

// Log struct for PufferLib
typedef struct {
    float perf;
    float score; // unnormalized score
    float n; // Required as the last field
    float episode_length; // Number of movements in the episode
    float episode_return; // Total reward in the episode
} Log;

typedef struct {
    int* disks; // Array to hold the disks on each peg
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

    Peg* pegs;

    Color* disk_colors; // Colors for each disk

    int moves;
    int size;
    bool won;

    int num_disks;
    int num_pegs;
} Hanoi;

// Add episode statistics to log
void add_log(Hanoi* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_length = env->moves;;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
}

float random_float(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

int random_int(int low, int high) {
    return low + rand() % (high - low + 1);
}

void init_disk_colors(Hanoi* env) {
    // Initialize colors for disks using random colors
    for(int i = 0; i < env->num_disks; i++) {
        env->disk_colors[i].r = (unsigned char)random_float(0, 255);
        env->disk_colors[i].g = (unsigned char)random_float(0, 255);
        env->disk_colors[i].b = (unsigned char)random_float(0, 255);
        env->disk_colors[i].a = 255; // Fully opaque
    }
}

void init(Hanoi* env) {
    // Initialize pegs
    env->pegs = (Peg*)malloc(sizeof(Peg) * env->num_pegs);
    for(int i = 0; i < env->num_pegs; i++) {
        env->pegs[i].disk_count = 0;
        env->pegs[i].disks = (int*)malloc(sizeof(int) * env->num_disks);
        for(int j = 0; j < env->num_disks; j++) {
            env->pegs[i].disks[j] = -1; // Initialize with -1 (no disks)
        }
    }

    // Place all disks in the first peg
    for(int i = 0; i < env->num_disks; i++) {
        int disk_idx = env->num_disks - i - 1; // From largest to smallest
        env->pegs[0].disks[i] = disk_idx;
        env->pegs[0].disk_count++;
    }

    // Initialize disk colors
    env->disk_colors = (Color*)malloc(sizeof(Color) * env->num_disks);
    init_disk_colors(env);
}

void c_reset(Hanoi* env) {
    // Clear pegs
    for(int i = 0; i < env->num_pegs; i++) {
        env->pegs[i].disk_count = 0;
        for(int j = 0; j < env->num_disks; j++) {
            env->pegs[i].disks[j] = -1; // Reset disks
        }
    }

    // Place all disks in the first peg
    for(int i = 0; i < env->num_disks; i++) {
        int disk_idx = env->num_disks - i - 1;
        env->pegs[0].disks[i] = disk_idx; // From largest to smallest
        env->pegs[0].disk_count++;
    }

    // First, reset observations to an "empty" value like -1
    for (int i = 0; i < env->num_pegs * env->num_disks; i++) {
        env->observations[i] = -1.0f;
    }

    // Then, fill in the current disk positions
    for (int i = 0; i < env->num_pegs; i++) { // For each peg
        for (int j = 0; j < env->pegs[i].disk_count; j++) { // For each disk on that peg
            int disk_id = env->pegs[i].disks[j];
            // The observation encodes the peg, the position in the stack, and the disk id
            env->observations[i * env->num_disks + j] = (float)disk_id;
        }
    }

    env->moves = 0;
    env->rewards[0] = 0.0f; // Reset reward
}

float compute_disks_reward(Hanoi* env) {
    // This method returns a reward based on the number of disks on the last peg
	Peg last_peg = env->pegs[env->num_pegs - 1];
	if(last_peg.disk_count == 0) {
		return 0.0f;
	}

	float reward = 0.0f;
	for(int i = 0; i < last_peg.disk_count; i++) {
		int disk_id = last_peg.disks[i];
		if(disk_id == env->num_disks - 1 - i) {
			reward += 1.0f;
		}
	}
	return reward;
}

void c_step(Hanoi* env) {

    int from_peg = env->actions[0];
    int to_peg = env->actions[1];

    env->moves++;

    if(env->pegs[from_peg].disk_count == 0) {
        // No disks to move from the selected peg
        env->rewards[0] = -1.0f; // Invalid action
        env->terminals[0] = 0;
        return;
    }

    // Get top disky
    int disk_to_move = env->pegs[from_peg].disks[env->pegs[from_peg].disk_count - 1];

    // If to_peg is not empty, check if move is valid (no larger disk on top)
    if(env->pegs[to_peg].disk_count > 0) {
        int top_disk = env->pegs[to_peg].disks[env->pegs[to_peg].disk_count - 1];
        if(disk_to_move > top_disk) {
            // Invalid move, larger disk on top
            env->rewards[0] = -1.0f; // Assign a smaller, but significant, penalty
            env->terminals[0] = 0;
            return;
        }
    }

	// Compute old state disks reward
	//float old_disks_reward = compute_disks_reward(env);

    // Move disk
    env->pegs[from_peg].disks[env->pegs[from_peg].disk_count - 1] = -1; // Remove disk from the peg
    env->pegs[from_peg].disk_count--;
    env->pegs[to_peg].disks[env->pegs[to_peg].disk_count] = disk_to_move;
    env->pegs[to_peg].disk_count++;

	// Compute new state disks reward
	//float new_disks_reward = compute_disks_reward(env);

    // Small penalty for each movel to encourage efficiency
    env->rewards[0] -= 0.1f;

    // We reward the agent by making less moves
    if (env->pegs[env->num_pegs - 1].disk_count == env->num_disks) {
        env->rewards[0] = 10.0f;
        env->terminals[0] = 1; // Episode is done

        add_log(env);
        c_reset(env);
        return;
    }

    if(env->moves >= MAX_MOVEMENTS) {
        env->rewards[0] = -10.0f; // Penalty for exceeding max movements
        env->terminals[0] = 1; // Episode is done
        add_log(env);
        c_reset(env);
        return;
    }

    // Update observations
    for (int i = 0; i < env->num_pegs * env->num_disks; i++) {
        env->observations[i] = -1.0f;
    }

    // Fill in the current disk positions
    for (int i = 0; i < env->num_pegs; i++) { // For each peg
        for (int j = 0; j < env->pegs[i].disk_count; j++) { // For each disk on that peg
            int disk_id = env->pegs[i].disks[j];
            // The observation encodes the peg, the position in the stack, and the disk id
            env->observations[i * env->num_disks + j] = (float)disk_id;
        }
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

    for (int i = 0; i < env->num_pegs; i++) {
        int peg_x = (SCREEN_WIDTH / (env->num_pegs + 1)) * (i + 1);
        DrawRectangle(peg_x - peg_width / 2, peg_y, peg_width, peg_height, GRAY);
    }

    // --- Draw Disks ---
    int disk_height = 25;
    for (int i = 0; i < env->num_pegs; i++) {
        int peg_x = (SCREEN_WIDTH / (env->num_pegs + 1)) * (i + 1);
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
    req.tv_nsec = 500 * 1000 * 1000; // 500 ms
    nanosleep(&req, NULL);
}