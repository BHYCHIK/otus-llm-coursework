from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from dotenv import load_dotenv
import os

from fastapi import FastAPI, Form, Response

from src.agents.graph import ReviewAnalyzer

load_dotenv('.env')

app = FastAPI()

review_analyzer = ReviewAnalyzer(os.environ.get("SENTIMENT_DETECTION_MODEL"))

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

    res = {
        'original_review': result.get('original_review'),
        'fixed_review': result.get('fixed_review'),
        'sentiment': result.get('sentiment'),
        'thread_id': 'thread_id',
        'review_fix_skipped': os.environ.get('SKIP_REVIEW_FIX') == 'true',
        'product_id': product_id,
        'good_points': {
            'speed_of_delivery': result.get('good_speed_of_delivery'),
            'price': result.get('good_price'),
            'quality': result.get('good_quality'),
            'good_looking': result.get('good_good_looking'),
            'fit_description': result.get('good_fit_description'),
            'functionality': result.get('good_functionality'),
        },
        'bad_points': {
            'speed_of_delivery': result.get('bad_speed_of_delivery'),
            'price': result.get('bad_price'),
            'quality': result.get('bad_quality'),
            'good_looking': result.get('bad_good_looking'),
            'fit_description': result.get('bad_fit_description'),
            'functionality': result.get('bad_functionality'),
        }
    }

    review_analysis_total.labels(
        sentiment=res['sentiment'],
        good_speed_of_delivery=res['good_points']['speed_of_delivery'],
        good_price=res['good_points']['price'],
        good_quality=res['good_points']['quality'],
        good_good_looking=res['good_points']['good_looking'],
        good_fit_description=res['good_points']['fit_description'],
        good_functionality=res['good_points']['functionality'],
        bad_speed_of_delivery=res['bad_points']['speed_of_delivery'],
        bad_price=res['bad_points']['price'],
        bad_quality=res['bad_points']['quality'],
        bad_good_looking=res['bad_points']['good_looking'],
        bad_fit_description=res['bad_points']['fit_description'],
        bad_functionality=res['bad_points']['functionality'],
    ).inc()

    return res