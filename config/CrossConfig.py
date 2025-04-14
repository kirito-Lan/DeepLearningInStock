from fastapi.middleware.cors import CORSMiddleware

from main import app

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # Allow all origins
    allow_credentials=True,    # When Allow credentials =Ture  allow_origins can't be set ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)