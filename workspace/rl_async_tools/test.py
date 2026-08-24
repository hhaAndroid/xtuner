from pydantic import BaseModel, Field


class BaseFoo(BaseModel):
    id: int


class FooA(BaseFoo):
    name: str


class FooB(BaseModel):
    a: BaseFoo



b = FooB(a=FooA(id=2, name="example"))
print(b.model_dump(mode="python"))