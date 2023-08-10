from datetime import datetime, timedelta
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

import database, models, settings

# Password hashing and verification context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for handling password bearer tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify the provided plain password against the hashed password.

    Parameters:
    - plain_password (str): The plain password to be verified.
    - hashed_password (str): The hashed password to compare against.

    Returns:
    - bool: True if the passwords match, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash the provided password.

    Parameters:
    - password (str): The plain password to be hashed.

    Returns:
    - str: The hashed password.
    """
    return pwd_context.hash(password)


def get_user(db: dict, username: str) -> models.UserInDB:
    """
    Retrieve a user from the database by their username.

    Parameters:
    - db (dict): The database of user information.
    - username (str): The username of the user to retrieve.

    Returns:
    - UserInDB: An instance of UserInDB if the user exists, None otherwise.
    """
    if username in db:
        user_dict = db[username]
        return models.UserInDB(**user_dict)


def authenticate_user(
    fake_db: dict, username: str, password: str
) -> Union[models.UserInDB, bool]:
    """
    Authenticate a user using their username and password.

    Parameters:
    - fake_db (dict): The database of user information.
    - username (str): The username of the user.
    - password (str): The plain password to verify.

    Returns:
    - Union[UserInDB, bool]: An instance of UserInDB if authentication is successful, False otherwise.
    """
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create an access token for the provided data.

    Parameters:
    - data (dict): The data to encode into the token.
    - expires_delta (Optional[timedelta]): Optional expiration time for the token.

    Returns:
    - str: The encoded JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )

    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> models.UserInDB:
    """
    Get the current user based on the provided JWT token.

    Parameters:
    - token (str): The JWT token extracted from the request header.

    Returns:
    - UserInDB: An instance of UserInDB representing the authenticated user.

    Raises:
    - HTTPException: If the token is invalid or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = models.TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(database.fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: models.UserInDB = Depends(get_current_user),
) -> models.UserInDB:
    """
    Get the current active user based on the provided authenticated user.

    Parameters:
    - current_user (UserInDB): The authenticated user obtained from the token.

    Returns:
    - UserInDB: An instance of UserInDB representing the current active user.

    Raises:
    - HTTPException: If the current user is disabled.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
