from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
import httpx

from .db import Base, engine, get_db
from .models import ReviewAnalysis
from .settings import settings
from .schemas import AnalyzerResponse

app = FastAPI(title="Review UI Service")

templates = Jinja2Templates(directory="./src/interface/templates")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request, "title": "Новый отзыв"})


@app.post("/submit")
async def submit_review(
    request: Request,
    review: str = Form(...),
    product_id: int = Form(...),
    db: Session = Depends(get_db),
):
    url = f"{settings.ANALYZER_BASE_URL.rstrip('/')}/analyze_review"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, data={"review": review, "product_id": product_id})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "title": "Новый отзыв",
                "error": f"Ошибка при запросе к analyzer-сервису: {e}",
            },
            status_code=502,
        )

    try:
        parsed = AnalyzerResponse.model_validate(data)
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "title": "Новый отзыв",
                "error": f"Analyzer вернул неожиданный формат: {e}",
            },
            status_code=502,
        )

    item = ReviewAnalysis(
        product_id=parsed.product_id,
        thread_id=parsed.thread_id,
        original_review=parsed.original_review,
        fixed_review=parsed.fixed_review,
        sentiment=parsed.sentiment,
        good_points=parsed.good_points,
        bad_points=parsed.bad_points,
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    # 4) Редирект на детальную страницу
    return RedirectResponse(url=f"/reviews/{item.id}", status_code=303)


@app.get("/reviews", response_class=HTMLResponse)
def list_reviews(request: Request, db: Session = Depends(get_db)):
    items = db.execute(select(ReviewAnalysis).order_by(desc(ReviewAnalysis.id))).scalars().all()
    return templates.TemplateResponse(
        request,
        "reviews.html",
        {"request": request, "title": "Все отзывы", "items": items},
    )


@app.get("/reviews/{review_id}", response_class=HTMLResponse)
def review_detail(review_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(ReviewAnalysis, review_id)
    if not item:
        return templates.TemplateResponse(
            request,
            "base.html",
            {"request": request, "title": "Не найдено", "error": "Отзыв не найден"},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "review_detail.html",
        {"request": request, "title": f"Отзыв #{item.id}", "item": item},
    )