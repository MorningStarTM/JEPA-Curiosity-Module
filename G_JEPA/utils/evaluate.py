import os
import numpy as np
from typing import Tuple, List, Dict, Optional
import json
from G_JEPA.utils.logger import logger
from G_JEPA.utils.converter import ActionConverter
from G_JEPA.trainer import ActorCriticTrainer




DATA_SPLIT = {
    "5": ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    "14": (
        list(range(0, 40 * 26, 40)),
        list(range(1, 100 * 10 + 1, 100)),
        list(range(2, 100 * 10 + 2, 100)),
    ),}



def compute_episode_score(env, ep_infos, chronic_id, dn_ffw, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7 # This defines the minimum acceptable ratio of losses to total demand for a good score.
        ep_marginal_cost = env.gen_cost_per_MW.max() # Maximum marginal cost of generation per MW in the environment.
        
        #  If ffw is None, retrieve episode information from ep_infos dictionary.
        if ffw is None: # If ffw is not provided, it means we're dealing with the full episode
            ep_do_nothing_reward = ep_infos[chronic_id]["donothing_reward"] # The total reward achieved by the "Do Nothing" agent for this chronic ID
            ep_do_nothing_nodisc_reward = ep_infos[chronic_id]["donothing_nodisc_reward"] # The non-discounted reward of the "Do Nothing" agent
            ep_dn_played = ep_infos[chronic_id]["dn_played"] # The number of steps the "Do Nothing" agent took actions during the episode for this chronic ID.
            ep_loads = np.array(ep_infos[chronic_id]["sum_loads"]) # A NumPy array containing the total energy loads for each step in the entire episode.
            ep_losses = np.array(ep_infos[chronic_id]["losses"]) # A NumPy array containing the energy losses for each step in the entire episode.
        
        # If ffw is provided, extract episode information for the specific fast-forward value from dn_ffw dictionary.
        else: # If ffw has a value, it indicates a fast-forward scenario where we only consider a specific portion of the episode.
            start_idx = 0 if ffw == 0 else ffw * 288 - 2 # This is set to 0 if ffw is 0 (no fast-forward). Otherwise, it calculates the starting index by multiplying ffw by the number of timesteps per day (288) and subtracting 2 (likely to account for potential edge effects).

            end_idx = start_idx + 864 # get the ending index for the specific fast-forward window.
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = dn_ffw[(chronic_id, ffw)] # Number of steps, Total reward & the Non - discounted reward y the "Do Nothing" agent for this chronic ID and fast-forward window. 
            # extracts sub-arrays of ep_loads and ep_losses using slicing based on the calculated start_idx and end_idx
            # This gives us the load and loss information only for the relevant fast-forward window within the episode.
            ep_loads = np.array(ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(ep_infos[chronic_id]["losses"])[start_idx:end_idx]


        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:] # Identifying Blackout Steps
        # Calculating Blackout Penalty
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost # Calculates the penalty for unmet demands during blackouts
            agent_reward += blackout_reward # Adds the calculated blackout penalty (blackout_reward) to the agent's total reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost # calculates the worst possible reward the agent could achieve
        best_reward = np.sum(ep_losses) * min_losses_ratio # calculates the best possible reward the agent could achieve 
        zero_reward = ep_do_nothing_reward # The "Do Nothing" agent represents a baseline performance where the agent takes no actions throughout the episode. Its reward serves as a reference point for evaluating the actual agent's performance.
        zero_blackout = ep_loads[ep_dn_played:] #  calculates the total penalty incurred by the "Do Nothing" agent due to unmet demands during blackouts.
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward # non-discounted reward of the "Do Nothing" agent for this specific scenario

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]

        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score


def get_max_ffw(case): # Retrieves value represents the maximum ffw for that specific case.
    MAX_FFW = {"5": 5, "14": 26, "118":7}
    return MAX_FFW[case]

def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(get_max_ffw(case)):
            with open(os.path.join(path, f"{i}_{j}.json"), "r", encoding="utf-8") as f:
                a = json.load(f)
                res[(i, j)] = (
                    a["dn_played"],
                    a["donothing_reward"],
                    a['donothing_nodisc_reward'],
                )
            #if i >= 2880:
            #    break
    return res


def select_chronics(env_path, case, eval=False):
    """_summary_
         This function orchestrates the selection and filtering of different power grid scenarios, 
        ensuring that the training and evaluation environments interact with appropriate chronics for their respective purposes. 
        It also extracts and organizes relevant information from those scenarios for use in the RL process.
    """
    
    # select chronics end define dn_ffw
    dn_json_path = os.path.join(env_path, "json")
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[case] # Retrieves pre-defined sets of chronics for training, validation, and testing based on the specific grid case (case) using a dictionary called DATA_SPLIT.
    ep_infos = {} # Initializes an empty dictionary to store information about chronic episodes.
    # Selects either test chronics (eval = True) or a combination of train and valid chronics (eval = False) for training or evaluation purposes.
    if eval:
        chronics = test_chronics
    else:
        chronics = train_chronics + valid_chronics
    # Calls a previously defined function to read and process information about "do nothing" actions and rewards from JSON files associated with the selected chronics.    
    dn_ffw = read_ffw_json(dn_json_path, chronics, case)

    if os.path.exists(dn_json_path): #  Checks if the JSON directory exists.
        for i in list(set(chronics)):
            with open(os.path.join(dn_json_path, f"{i}.json"), "r", encoding="utf-8") as f:
                ep_infos[i] = json.load(f) # Iterates through unique chronics and loads additional information about each episode from its corresponding JSON file into the ep_infos dictionary.

    return dn_ffw, ep_infos




def evaluate_agent(
    agent,
    test_env,
    json_path: str,
    chronics: List,
    trainer:ActorCriticTrainer,
    max_ffw: int = 26,  # Default for case 14
    path: str = "./results",
    sample: bool = False,
    compute_episode_score_fn=compute_episode_score,
) -> Tuple[Dict, List, List]:
    """
    Evaluates a trained RL agent across multiple power grid scenarios (chronics).
    
    Args:
        agent: Trained RL agent with act() and reset() methods
        test_env: Grid2op evaluation environment
        dn_ffw: Dictionary with "Do Nothing" agent performance data
                Format: {(chronic_id, ffw): (steps_survived, reward, nodisc_reward)}
        ep_infos: Dictionary with episode information for each chronic
                  Format: {chronic_id: {...episode_data...}}
        chronics: List of chronic IDs to evaluate on
        max_ffw: Maximum fast-forward weeks (default 26 for case 14)
        path: Directory to save evaluation results
        sample: If True, sample actions; if False, use deterministic actions
        compute_episode_score_fn: Function to compute L2RPN score
                                  Signature: fn(chronic_id, alive_frame, total_reward, ffw)
    
    Returns:
        Tuple of (stats_dict, scores_list, steps_list)
        - stats_dict: {"step": avg_steps, "score": avg_score, "reward": avg_reward}
        - scores_list: L2RPN scores for each chronic
        - steps_list: Steps survived for each chronic
    """
    
    result = {}
    lenOfChronics = []
    steps, scores = [], []
    ac_converter = ActionConverter(test_env)
    
    dn_ffw, ep_infos = select_chronics(env_path=json_path, case="14", eval=True)
    logger.info(f"dn_ffw loaded with {len(dn_ffw)} entries.")
    # Create output directory if sampling
    if sample:
        sample_path = os.path.join(path, "sample/")
        os.makedirs(sample_path, exist_ok=True)
        path = sample_path
    else:
        os.makedirs(path, exist_ok=True)

    # Handle case where max_ffw == 5 (replicate chronics)
    if max_ffw == 5:
        chronics = chronics * 5

    for idx, chron_id in enumerate(chronics):
        # Determine fast-forward window
        if max_ffw == 5:
            ffw = idx % 5
        else:
            # Select the hardest fast-forward window (where DN agent survived least)
            # ffw = int(
            #     np.argmin(
            #         [
            #             dn_ffw[(chron_id, fw)][0]
            #             for fw in range(max_ffw)
            #             if (chron_id, fw) in dn_ffw and dn_ffw[(chron_id, fw)][0] >= 7
            #         ]
            #     )
            # )
            # print(f"FFW selected for chronic {chron_id}: {ffw}")
            candidates = [
                (fw, dn_ffw[(chron_id, fw)][0])
                for fw in range(max_ffw)
                if (chron_id, fw) in dn_ffw and dn_ffw[(chron_id, fw)][0] >= 7
            ]

            if not candidates:
                ffw = 0
            else:
                ffw = min(candidates, key=lambda x: x[1])[0]  # fw with smallest dn_step

            print(f"FFW selected for chronic {chron_id}: {ffw}")
        # Get baseline performance
        dn_step = dn_ffw[(chron_id, ffw)][0]
        
        # Initialize environment
        test_env.seed(59)
        test_env.set_id(chron_id)
        obs = test_env.reset()
        #agent.reset(obs)

        def is_safe(obs):
            for ratio, limit in zip(obs.rho, test_env._thermal_limit_a):
                # Seperate big line and small line
                if (limit < 400.00 and ratio >= 0.9 - 0.05) or ratio >= 0.9:
                    return False
            return True

        # Fast-forward if needed
        if ffw > 0:
            test_env.fast_forward_chronics(ffw * 288 - 3)
            obs, *_ = test_env.step(test_env.action_space())

        # Initialize tracking variables
        total_reward = 0
        alive_frame = 0
        done = False
        topo_changes = np.array([obs.topo_vect])
        safe = np.array([1])
        result[(chron_id, ffw)] = {}

        # Run episode
        while not done:
            # Get agent action
            safe_check = is_safe(obs)
            sub_id = trainer.pick_sub_rule_based(np.array([0,1,2,3,4,5,6,8,9,10,11,12,13]), test_env.action_space, obs)
            with_neighbors, _ = trainer.get_connected_substations(env=test_env, sub_id=sub_id, return_line_ids=True)
            allowed_actions = trainer.find_action_ids(sub_id=with_neighbors)
            act = agent.act_top_k(obs.to_vect(), allowed_action_ids=allowed_actions)
            grid_action = ac_converter.act(act)
            # if safe_check:
            #     act = agent(obs.to_vect())
            #     grid_action = ac_converter.act(act)
            # else:
            #     grid_action = test_env.action_space({})
            
            # Track topology changes and safety
            prev_topo = obs.topo_vect
            topo_changes = np.vstack((topo_changes, prev_topo))
            safe = np.append(safe, is_safe(obs))
            
            # Step environment
            obs, reward, done, info = test_env.step(grid_action)
            total_reward += reward
            alive_frame += 1
            
            # Episode limit (3 days = 864 timesteps)
            if alive_frame == 864:
                done = True

        # Compute score
        if compute_episode_score_fn is not None:
            l2rpn_score = float(compute_episode_score(test_env, ep_infos, chron_id, dn_ffw, alive_frame, total_reward, ffw=ffw))
            #env, ep_infos, chronic_id, dn_ffw, agent_step, agent_reward
        else:
            # Fallback: return raw reward if no scoring function provided
            l2rpn_score = float(total_reward)

        # Save trajectory data
        np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_topo.npy"), topo_changes)
        np.save(os.path.join(path, f"Ch{chron_id}_{ffw}_safe.npy"), safe)

        # Log results
        print(
            f"[Test Ch{chron_id}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) "
            f"Score: {l2rpn_score:9.4f}"
        )

        # Store results
        lenOfChronics.append(chron_id)
        scores.append(l2rpn_score)
        steps.append(alive_frame)
        result[(chron_id, ffw)]["real_reward"] = total_reward
        result[(chron_id, ffw)]["reward"] = l2rpn_score
        result[(chron_id, ffw)]["step"] = alive_frame

    # Compute statistics
    val_step = val_score = val_rew = 0
    for key in result:
        val_step += result[key]["step"]
        val_score += result[key]["reward"]
        val_rew += result[key]["real_reward"]

    stats = {
        "step": val_step / len(chronics),
        "score": val_score / len(chronics),
        "reward": val_rew / len(chronics),
    }
    
    tot_score = np.array(scores)
    avg = np.sum(tot_score) / len(lenOfChronics)
    print(
        f"\nEvaluation Summary:"
        f"\n  Total Score: {np.sum(tot_score):.2f}"
        f"\n  Num Chronics: {len(lenOfChronics)}"
        f"\n  Average Score: {avg:.4f}"
    )

    return stats, scores, steps