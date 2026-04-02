from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from dotenv import load_dotenv
import os

from fastapi import FastAPI, Form, Response

from src.agents.graph import ReviewAnalyzer

load_dotenv('.env')

app = FastAPI()

review_analyzer = ReviewAnalyzer(sentiment_detection_model=os.environ.get("SENTIMENT_DETECTION_MODEL"),
                                 skip_review_fix=os.environ.get("SKIP_REVIEW_FIX")=="true",
                                 llm_base_url=os.environ.get("BASEURL"),
                                 llm_api_key=os.environ.get("APIKEY"))

review_analysis_total = Counter(
    "review_analysis_total",
    "Total review analysis",
    ['sentiment',
     'good_speed_of_delivery', 'good_price', 'good_quality', 'good_good_looking', 'good_fit_description',
     'good_functionality',
     'bad_speed_of_delivery', 'bad_price', 'bad_quality', 'bad_good_looking', 'bad_fit_description',
     'bad_functionality'],
)
@app.get('/')
def root():
    return {'status': 'ok'}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/analyze_review')
def analyze_review(review: str = Form(..., description="Review from marketplace"),
                   product_id: int = Form(..., description="Product ID")):

    result = review_analyzer.analyze(review)
    result['product_id'] = product_id

    review_analysis_total.labels(
        sentiment=result['sentiment'],
        good_speed_of_delivery=result['good_points']['speed_of_delivery'],
        good_price=result['good_points']['price'],
        good_quality=result['good_points']['quality'],
        good_good_looking=result['good_points']['good_looking'],
        good_fit_description=result['good_points']['fit_description'],
        good_functionality=result['good_points']['functionality'],
        bad_speed_of_delivery=result['bad_points']['speed_of_delivery'],
        bad_price=result['bad_points']['price'],
        bad_quality=result['bad_points']['quality'],
        bad_good_looking=result['bad_points']['good_looking'],
        bad_fit_description=result['bad_points']['fit_description'],
        bad_functionality=result['bad_points']['functionality'],
    ).inc()

    return result