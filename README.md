# Robust PIBT Pathfinding Algorithm for Multi-Agent Systems

An Honours research project investigating improvements to Priority Inheritance with Backtracking (PIBT), a priority-based algorithm for multi-agent pathfinding.

## Overview

Multi-agent pathfinding (MAPF) involves coordinating multiple agents navigating a shared environment while avoiding collisions and reaching their respective destinations.

Priority Inheritance with Backtracking (PIBT) is a decentralised approach that assigns priorities to agents and uses backtracking to resolve movement conflicts.

However, certain configurations can cause agents to become trapped in deadlocks or repeatedly move without making meaningful progress, resulting in livelocks.

This project investigates ways to improve the robustness of PIBT by identifying these failure cases and implementing additional strategies to help agents continue navigating the environment.

## Research Objectives

The main objectives were to:

- Investigate the behaviour and limitations of the baseline PIBT algorithm.
- Identify scenarios that lead to deadlocks and livelocks.
- Develop additional strategies to address these failure cases.
- Evaluate the effectiveness of the proposed improvements through simulation and experimentation.

## Implementation

**Language:** Python

The algorithm was implemented and evaluated in a 2D grid environment with the following assumptions:

- Static environment with fixed obstacles.
- Synchronous agent movement.
- Homogeneous agents with the same movement capabilities.
- Multiple agents navigating the same grid.

### Baseline PIBT

The baseline algorithm uses priority inheritance and backtracking to resolve movement conflicts between agents.

Each agent is assigned a priority, which determines the order in which movement decisions are considered. When an agent cannot move to its preferred location, priority inheritance and backtracking are used to explore alternative movements.

### Deadlock Resolution

Deadlocks occur when agents block one another and cannot make progress.

The project investigates blocking relationships between agents and introduces additional resolution strategies, including:

- Push-and-Swap strategies.
- Corridor swap handling.
- Frozen priority mechanisms.

These approaches aim to resolve situations where agents become mutually blocked.

### Livelock Resolution

Livelocks occur when agents continue moving but fail to make meaningful progress towards their destinations.

A push-counter matrix was introduced to track repeated pushing behaviour and help identify problematic movement patterns.

This information was used to guide additional movement decisions and reduce repeated unproductive interactions.

## Evaluation

The project involved testing the baseline and modified algorithms across different grid configurations and agent scenarios.

The experiments focused on:

- Identifying deadlock and livelock scenarios.
- Observing agent movement and algorithm behaviour.
- Comparing different resolution strategies.
- Evaluating the effectiveness of the proposed improvements.

## Technologies

- **Language:** Python
- **Domain:** Algorithms and Data Structures
- **Research Areas:** Multi-Agent Pathfinding, Graph Search, Deadlock Detection, Algorithm Optimisation

## Repository Contents

The repository contains the implementation, experimental materials and supporting research documentation.

*Update this section with the final folder structure if the repository contains separate source, experiment and documentation directories.*

## Scope and Limitations

This research focuses on a controlled 2D grid environment with static obstacles, synchronous movement and homogeneous agents.

The findings are therefore evaluated within this defined environment and do not establish performance across all MAPF environments or more complex real-world scenarios.

## Research Documentation

For the complete methodology, implementation details, experiments and findings, refer to the Honours thesis:

**[PIBT Honours Thesis](./public/thesis/PIBT-Thesis.pdf)**

## Author

**Samantha Oh**

Bachelor of Computer Science (Honours)  
Monash University
