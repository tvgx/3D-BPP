import yaml
import os

class ConfigLoader:
    """
    Utility to load and merge configuration files.
    Allows overriding default settings with specific algorithm or simulator configs.
    """
    
    @staticmethod
    def load_config(file_path):
        try:
            # Thêm encoding='utf-8' vào hàm open
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except UnicodeDecodeError:
            # Backup cho trường hợp file lưu ở định dạng khác
            with open(file_path, 'r', encoding='latin-1') as f:
                return yaml.safe_load(f) or {}

    @staticmethod
    def merge_configs(base_config, override_config):
        """
        Merges override_config into base_config recursively.
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                ConfigLoader.merge_configs(base_config[key], value)
            else:
                base_config[key] = value
        return base_config

    @staticmethod
    def get_algorithm_config(algo_name, config_dir="config"):
        """
        Loads the specific config for an algorithm name.
        Mapping:
        - ga -> ga_classic.yaml
        - brkga -> brkga_v2.yaml
        - aco -> aco_pheromone.yaml
        - pso -> pso_swarm.yaml
        - de -> de_shade.yaml
        - es -> es_cma_es.yaml
        - mfea -> mfea_multitask.yaml
        - nsga2 -> nsga2_multi_obj.yaml
        """
        mapping = {
            "ga": "ga_classic.yaml",
            "brkga": "brkga_v2.yaml",
            "de": "de_shade.yaml",
            "es": "es_cma_es.yaml",
            "cmaes": "es_cma_es.yaml",
            "mfea": "mfea_multitask.yaml",
            "nsga2": "nsga2_multi_obj.yaml"
        }
        
        filename = mapping.get(algo_name.lower())
        if not filename:
            print(f"Warning: No specific config map for {algo_name}. Using default.")
            return {}
            
        return ConfigLoader.load_config(os.path.join(config_dir, filename))
