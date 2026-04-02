from typing import TypedDict, NotRequired

class State(TypedDict):
    original_review: str
    fixed_review: NotRequired[str]
    sentiment: NotRequired[str]

    good_speed_of_delivery: NotRequired[bool]
    good_price: NotRequired[bool]
    good_quality: NotRequired[bool]
    good_good_looking: NotRequired[bool]
    good_fit_description: NotRequired[bool]
    good_functionality: NotRequired[bool]

    bad_speed_of_delivery: NotRequired[bool]
    bad_price: NotRequired[bool]
    bad_quality: NotRequired[bool]
    bad_good_looking: NotRequired[bool]
    bad_fit_description: NotRequired[bool]
    bad_functionality: NotRequired[bool]