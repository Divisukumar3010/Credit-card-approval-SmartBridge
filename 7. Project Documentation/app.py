from flask import Flask, render_template, request
import os
import pickle
import hashlib
import random

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    bundle = pickle.load(file)

REQUIRED_KEYS = {
    'models', 'gender_encoder', 'own_car_encoder', 'own_realty_encoder',
    'income_type_encoder', 'education_type_encoder', 'family_status_encoder',
    'housing_type_encoder', 'occupation_type_encoder', 'feature_order',
}

if not isinstance(bundle, dict) or not REQUIRED_KEYS.issubset(bundle.keys()):
    raise RuntimeError(
        "model.pkl is in the old format (missing one or more encoders). "
        "Re-run the updated main.ipynb from top to bottom to regenerate model.pkl "
        "as a bundle containing 'models', 'feature_order', and an encoder for every "
        "categorical field the form now collects: gender, own_car, own_realty, "
        "income_type, education_type, family_status, housing_type, occupation_type."
    )

loaded_models = bundle['models']
feature_order = bundle['feature_order']

# One LabelEncoder per categorical form field, keyed by the field name used below.
ENCODERS = {
    'gender': bundle['gender_encoder'],
    'own_car': bundle['own_car_encoder'],
    'own_realty': bundle['own_realty_encoder'],
    'income_type': bundle['income_type_encoder'],
    'education_type': bundle['education_type_encoder'],
    'family_status': bundle['family_status_encoder'],
    'housing_type': bundle['housing_type_encoder'],
    'occupation_type': bundle['occupation_type_encoder'],
}

# Maps a FEATURE_ORDER column name to the form field that feeds it.
ENCODED_FEATURE_TO_FIELD = {
    'GENDER_ENC': 'gender',
    'OWN_CAR_ENC': 'own_car',
    'OWN_REALTY_ENC': 'own_realty',
    'INCOME_TYPE_ENC': 'income_type',
    'EDUCATION_TYPE_ENC': 'education_type',
    'FAMILY_STATUS_ENC': 'family_status',
    'HOUSING_TYPE_ENC': 'housing_type',
    'OCCUPATION_TYPE_ENC': 'occupation_type',
}

model_name = os.environ.get('MODEL_NAME', 'random_forest')
model = loaded_models.get(model_name, next(iter(loaded_models.values())))


def encode_category(field_name, value):
    """Turn a form string into the numeric code the model was trained on.
    Falls back to the most common training category if the form sends a
    value the encoder has never seen, instead of crashing.
    """
    encoder = ENCODERS[field_name]
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    fallback = encoder.classes_[0]
    print(f"Warning: unseen {field_name} '{value}', falling back to '{fallback}'")
    return int(encoder.transform([fallback])[0])


def to_float(raw, default=0.0):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def to_int(raw, default=0):
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def luhn_check_digit(digits):
    """Given all digits except the last, compute the final digit that makes
    the full number pass the Luhn checksum real card numbers use."""
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10


def generate_card_number(seed_text):
    """Deterministic per-applicant 16-digit number, grouped like a real card,
    with a genuine Luhn check digit so it 'feels' like a real card number."""
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    prefix = rng.choice(['4', '51', '52', '53', '54', '55'])  # Visa- / Mastercard-style ranges
    digits = [int(d) for d in prefix]
    while len(digits) < 15:
        digits.append(rng.randint(0, 9))
    digits.append(luhn_check_digit(digits))
    number = ''.join(str(d) for d in digits)
    return ' '.join(number[i:i + 4] for i in range(0, 16, 4))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    applicant_name = request.form.get('applicant_name', 'Applicant')
    income_type = request.form.get('income_type', '')
    education_type = request.form.get('education_type', '')
    annual_income_raw = request.form.get('annual_income', '')

    try:
        annual_income = float(annual_income_raw)
    except (TypeError, ValueError):
        return render_template(
            'result.html',
            prediction_text='Please enter a numeric annual income.',
            approved=False,
            applicant_name=applicant_name,
            card_number=None,
        )

    # Remaining applicant fields. Each has a sane default so the app never 500s
    # even if a field is missing from the submitted form.
    gender = request.form.get('gender', 'F')
    own_car = request.form.get('own_car', 'N')
    own_realty = request.form.get('own_realty', 'N')
    family_status = request.form.get('family_status', 'Single / not married')
    housing_type = request.form.get('housing_type', 'House / apartment')
    occupation_type = request.form.get('occupation_type', 'Unknown')
    cnt_children = to_int(request.form.get('cnt_children'), 0)
    cnt_fam_members = to_int(request.form.get('cnt_fam_members'), max(cnt_children + 1, 1))
    age_years = to_float(request.form.get('age'), 30)
    years_employed = to_float(request.form.get('years_employed'), 0)

    raw_values = {
        'gender': gender,
        'own_car': own_car,
        'own_realty': own_realty,
        'income_type': income_type,
        'education_type': education_type,
        'family_status': family_status,
        'housing_type': housing_type,
        'occupation_type': occupation_type,
    }

    numeric_values = {
        'CNT_CHILDREN': cnt_children,
        'CNT_FAM_MEMBERS': cnt_fam_members,
        'AGE_YEARS': age_years,
        'YEARS_EMPLOYED': years_employed,
        'AMT_INCOME_TOTAL': annual_income,
    }

    # Build the feature vector in the exact order used at training time
    feature_values = dict(numeric_values)
    for enc_col, field_name in ENCODED_FEATURE_TO_FIELD.items():
        feature_values[enc_col] = encode_category(field_name, raw_values[field_name])

    final_input = [[feature_values[col] for col in feature_order]]

    prediction = model.predict(final_input)
    approved = bool(prediction[0] == 1)
    result = "Credit Card Approved" if approved else "Credit Card Rejected"

    # Only an approved application gets an actual card number issued.
    card_number = None
    if approved:
        seed_text = "|".join([
            applicant_name, gender, income_type, education_type, family_status,
            housing_type, occupation_type, str(annual_income), str(age_years),
        ])
        card_number = generate_card_number(seed_text)

    return render_template(
        'result.html',
        prediction_text=result,
        approved=approved,
        applicant_name=applicant_name,
        income_type=income_type,
        education_type=education_type,
        annual_income=int(annual_income),
        card_number=card_number,
    )


if __name__ == "__main__":
    app.run(debug=True)
