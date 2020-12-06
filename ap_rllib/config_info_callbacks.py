from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np


class InfoCallbacks(DefaultCallbacks):
    """Add data from episode infos to trainer results."""
    
    # fields in infos to look for
    # format: field name, player, mode (aggregate over episodes)
    FIELDS = {'contact': ('player_1', np.max)}
    
    def get_info(self, episode):
        """Save info for all players from the last step."""
        info = {p: episode.last_info_for(p) for p in episode.user_data["policies"]}
        return info
    
    def on_episode_start(self, *, worker, base_env, policies,
                         episode, env_index, **kwargs):
        """Save the policies list and reset the info metrics."""
        episode.user_data["policies"] = policies
        episode.user_data["info"] = []
        episode.user_data["info"].append(self.get_info(episode))
        
    def on_episode_step(self, *, worker, base_env,
                         episode, env_index, **kwargs):
        """Add info metrics."""
        episode.user_data["info"].append(self.get_info(episode))

    def on_episode_end(self, *, worker, base_env, policies,
                         episode, env_index, **kwargs):
        """Report aggregate metrics."""
        episode.user_data["info"].append(self.get_info(episode))
        
        # reporting data
        for key, (player, mode) in self.FIELDS.items():
            # array of infos
            data = episode.user_data.get("info", [])
            
            # array of infos for the player
            data = [step.get(player, {}) for step in data if step is not None]
            
            # array of values for the player
            data = [info.get(key, None) for info in data if info is not None]
            data = [x for x in data if x is not None]
            episode.custom_metrics["contacts"] = mode(data)