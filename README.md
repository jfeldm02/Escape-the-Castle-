# Escape-the-Castle

A manual application (no AI libraries) of Markov Decision Processes and reinforcement learning to teach an agent to successfully navigate a grid from a source to a goal state while interacting.

## Overview
In this assignment, I used my knowledge of MDPs and reinforcement learning to play a game, *Escape the Castle*. The game is set on a 5x5 grid, where the player’s goal is to navigate from the top-left corner (grid position `(0,0)`) to the bottom-right corner (grid position `(4,4)`). The environment is populated with four hidden guards (whose positions are randomly initialized on every run). Each guard possesses distinct attributes, such as their strength in combat or perception ability in detecting players. The player can only discover a guard’s presence by entering the same cell as the guard. The game dynamics are governed by an underlying Markov Decision Process (MDP) with probabilistic movement and guard interactions.

## Grid Layout
- The grid consists of 25 cells, numbered from `(0,0)` to `(4,4)`.
- The player always starts in the top-left corner `(0,0)`.
- The goal is located in the bottom-right corner `(4,4)`.
- Four guards are randomly positioned across the grid at the start of each game, excluding the goal and starting positions. The player does not know the guards’ locations until they encounter one in a cell.

## Player and Guard Interactions
- When the player enters a cell with a guard, they are faced with two options: **hide** or **fight**. Movement actions are invalid in this phase.
- If the player fails to hide, they are forced into combat with the guard. The environment automatically executes a fight sequence and returns the outcome and associated reward/penalty.
- Each of the guards has distinct, hidden strengths (for combat) and perception abilities (for detecting hiding attempts). The player is unaware of these characteristics before engaging with each guard.
- The player’s interaction with a guard results in one of the following outcomes:
  - **Victory in combat**: The player wins the fight, is moved randomly to a neighboring cell, and receives a reward.
  - **Defeat in combat**: The player loses, suffers a reduction in health, is moved to a neighboring cell, and receives a penalty.
  - **Successful hiding**: The player successfully avoids combat and is moved to a neighboring cell.
  - **Failed hiding**: The player fails to hide and is forced to fight.
- Regardless of the outcome, guard positions remain unchanged throughout the game.

## Player’s State
The player’s current state is represented by three key factors:
1. **Position on the grid**: The player can be in any of the 25 cells.
2. **Health status**: The player has three possible health states—`Full`, `Injured`, and `Critical`. Losing combat reduces health. If health reaches the `Critical` state and the player loses another combat, the game ends in defeat.
3. **Guard positions**: The four guards are randomly positioned at the start of each game. The number of possible guard placements is combinatorial, resulting in `23C4` potential configurations, excluding the player’s starting and goal cells.

## Available Actions
The player can take one of six possible actions during the game: 

### Movement:
- **UP**
- **DOWN**
- **LEFT**
- **RIGHT**

### Interaction:
- **HIDE**
- **FIGHT**

#### Movement Details:
- The player can move to any adjacent cell on the grid (up, down, left, or right) with a 90% success rate. The remaining 10% represents a “slip,” where the player is moved to a random adjacent cell instead.
- Movement actions are not allowed when the player is in the same cell as a guard, forcing the player to either fight or hide. 
- Attempting to fight or hide in an empty cell results in no effect.

#### Interaction Details:
- **HIDE**: The player attempts to evade the guard. The success of this action is determined by the guard’s keenness level.
- **FIGHT**: The player engages in combat with the guard. The outcome of the fight is determined by the guard’s strength.

## Rewards and Penalties
The player receives rewards and penalties based on their actions and outcomes:
- **Reaching the Goal**: +10,000 reward points for successfully reaching the goal at grid `(4,4)`.
