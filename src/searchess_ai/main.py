from searchess_ai.api.app import create_app
import uvicorn

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "searchess_ai.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )