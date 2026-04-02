from pydantic import BaseModel, Field

class PointsOfReview(BaseModel):
    good_speed_of_delivery: bool = Field(
        description='Пользователю нравится скорость доставки. Если в тексте нет явного упоминания — ставь false')
    good_price: bool = Field(
        description='Пользователю нравится цена. Если в тексте нет явного упоминания — ставь false')
    good_quality: bool = Field(
        description='Пользователю нравится качество товара. Если в тексте нет явного упоминания — ставь false')
    good_good_looking: bool = Field(
        description='Пользователю нравится дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    good_fit_description: bool = Field(
        description='Пользователю нравится то, что товар соответствует описанию товара. Если в тексте нет явного упоминания — ставь false')
    good_functionality: bool = Field(
        description='Пользователю нравится функциональность товара. Если в тексте нет явного упоминания — ставь false')

    bad_speed_of_delivery: bool = Field(
        description='Пользователя расстраивает скорость доставки. Если в тексте нет явного упоминания — ставь false')
    bad_price: bool = Field(
        description='Пользователя расстраивает цена. Если в тексте нет явного упоминания — ставь false')
    bad_quality: bool = Field(
        description='Пользователя расстраивает качество товара. Если в тексте нет явного упоминания — ставь false')
    bad_good_looking: bool = Field(
        description='Пользователя расстраивает дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    bad_fit_description: bool = Field(
        description='Пользователя расстраивает то, что товар отличается от описания товара. Если в тексте нет явного упоминания — ставь false')
    bad_functionality: bool = Field(
        description='Пользователя расстраивает функциональность товара. Если в тексте нет явного упоминания — ставь false')