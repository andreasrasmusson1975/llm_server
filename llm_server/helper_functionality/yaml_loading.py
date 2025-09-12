import yaml
from pathlib import Path

def load_yaml(yaml_file_name: str) -> dict:
    """
    Load and parse a YAML file from the project's 'yaml' directory.

    Resolves the file path relative to the project root, opens the YAML file, and returns its contents as a Python dictionary.
    Uses safe loading to avoid executing arbitrary code in the YAML file.

    Args:
        yaml_file_name (str): The name of the YAML file to load (e.g., 'model_config.yaml').

    Returns:
        dict: The parsed contents of the YAML file as a dictionary.
    """
    path = Path(__file__).parent.parent / "yaml_files" / yaml_file_name
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model_config() -> dict:
    """
    Load the model configuration from the 'model_config.yaml' file.

    This function retrieves model-specific configuration parameters from the YAML file located in the project's 'yaml' directory.
    It is typically used to centralize model settings for LLM pipelines and related applications.

    Returns:
        dict: The parsed model configuration as a dictionary.
    """
    return load_yaml("model_config.yaml")["model"]

def load_system_prompt() -> str:
    """
    Load the system prompt string from the 'system_prompt_config.yaml' file.

    This function retrieves the system prompt used for initializing or guiding LLM behavior from the YAML configuration file
    located in the project's 'yaml' directory. It returns the value associated with the 'system_prompt' key, or an empty string if not found.

    Returns:
        str: The system prompt string for the LLM, or an empty string if not present in the config.
    """
    data = load_yaml("system_prompt_config.yaml")
    return data.get("system_prompts", "")

def load_chat_server_config() -> dict:
    """
    Load the chat server configuration from the 'server_config.yaml' file.

    This function retrieves server-specific configuration parameters such as host, port, and API key
    from the YAML file located in the project's 'yaml' directory. It is typically used to configure
    the connection settings for a chat server.

    Returns:
        dict: The parsed server configuration as a dictionary.
    """
    return load_yaml("server_config.yaml").get("server", {})

def load_azure_config() -> dict:
    """
    Load the Azure configuration from the 'azure_config.yaml' file.

    This function retrieves Azure-specific configuration parameters such as key vault name and secret name
    from the YAML file located in the project's 'yaml' directory. It is typically used to configure
    access to Azure services.

    Returns:
        dict: The parsed Azure configuration as a dictionary.
    """
    return load_yaml("azure_config.yaml").get("azure", {})
