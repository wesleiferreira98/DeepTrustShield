from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_admin, get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db.database import get_db
from app.db.models import User

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreate(BaseModel):
    email: str
    name: str
    password: str
    is_admin: bool = False


def _fmt(user: User) -> dict:
    return {
        "id":         user.id,
        "email":      user.email,
        "name":       user.name,
        "is_admin":   user.is_admin,
        "is_active":  user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email ou senha incorretos.")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Conta desativada. Contate o administrador.")

    token, expires_in = create_access_token(user.id, user.email)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   expires_in,
        "user":         _fmt(user),
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return _fmt(current_user)


@router.get("/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(get_current_admin)):
    return [_fmt(u) for u in db.query(User).order_by(User.created_at).all()]


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin),
):
    email = body.email.lower().strip()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email já cadastrado.")

    user = User(
        email=email,
        name=body.name.strip(),
        hashed_password=hash_password(body.password),
        is_admin=body.is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _fmt(user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="Não é possível excluir a própria conta.")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    db.delete(user)
    db.commit()
