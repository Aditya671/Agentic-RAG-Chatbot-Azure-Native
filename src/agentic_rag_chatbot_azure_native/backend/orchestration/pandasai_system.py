from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI as PandasAIAzureOpenAI

# ... inside AsyncAgenticCSVChatEngine class ...

def __build_csv_engine(self):
    """
    Replaces the legacy PandasQueryEngine with PandasAI SmartDataframe.
    """
    # 1. Load the data using your existing utility [cite: 104, 105]
    df, meta = self.load_csv_file(self.blob_bytes['bytes'], self.blob_bytes['metadata'])
    
    # 2. Configure the PandasAI LLM wrapper using your credential manager [cite: 177]
    # Note: PandasAI requires its own LLM instance wrapper
    llm = PandasAIAzureOpenAI(
        api_token=self.credential_manager.get_secret('aoai-api-key'),
        azure_endpoint=self.config.llms.get('aoai').get('endpoint-east-us-2'),
        api_version=self.config.llms.get('aoai').get('api-version-east-us-2'),
        deployment_name=self.selected_model.value
    )
    
    # 3. Create the SmartDataframe
    # This replaces the entire PandasQueryEngine logic 
    try:
        engine = SmartDataframe(df, config={"llm": llm, "verbose": True})
        logger.info("[AgenticAi] PandasAI SmartDataframe created successfully")
        return engine
    except Exception as e:
        logger.error(f"[AgenticAi] Failed to create PandasAI engine: {str(e)}")
        raise