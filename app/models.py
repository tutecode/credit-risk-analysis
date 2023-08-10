from pydantic import BaseModel
from typing import Optional

# Pydantic model for representing an access token
class Token(BaseModel):
    access_token: str
    token_type: str

# Pydantic model for token data (username)
class TokenData(BaseModel):
    username: Optional[str] = None

# Pydantic model for user information
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

# Pydantic model for user information stored in the database
class UserInDB(User):
    hashed_password: str
