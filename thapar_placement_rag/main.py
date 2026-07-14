import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import snowflake.connector
from contextlib import asynccontextmanager

# --- Configuration ---
# It's best practice to set these as environment variables
SF_USER = os.getenv("SF_USER", "your_username")
SF_PASSWORD = os.getenv("SF_PASSWORD", "your_password")
SF_ACCOUNT = os.getenv("SF_ACCOUNT", "your_account_locator")
SF_WAREHOUSE = os.getenv("SF_WAREHOUSE", "COMPUTE_WH")
SF_DATABASE = os.getenv("SF_DATABASE", "THAPAR")
SF_SCHEMA = os.getenv("SF_SCHEMA", "PUBLIC")
SF_ROLE = os.getenv("SF_ROLE", "ACCOUNTADMIN")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- Snowflake Connection Helper ---
def get_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=SF_USER,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,
            warehouse=SF_WAREHOUSE,
            database=SF_DATABASE,
            schema=SF_SCHEMA,
            role=SF_ROLE
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Snowflake: {str(e)}")

# --- FastAPI App ---
app = FastAPI(title="Thapar Placement RAG API", description="API to query the Snowflake Cortex RAG engine.")

@app.post("/chat", response_model=QueryResponse)
async def chat_with_document(request: QueryRequest):
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        
        # We use %s binding to safely pass the user's question to the stored procedure
        sql_command = "CALL ask_cortex_rag(%s)"
        
        # Execute the call
        cursor.execute(sql_command, (request.question,))
        result = cursor.fetchone()
        
        if result:
            return QueryResponse(answer=result[0])
        else:
            raise HTTPException(status_code=500, detail="No response returned from Snowflake.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG query: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/health")
async def health_check():
    return {"status": "API is running natively against Snowflake Cortex."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)