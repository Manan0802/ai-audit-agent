
# definitions.py
# Hardcoded definitions for the 12 QA parameters.
# These are injected into the Gemini prompt so the AI understands
# what it is verifying against for each field.

PARAMETER_DEFINITIONS = {
    "call_type": (
        "Indicates whether the conversation is B2B (Business-to-Business) or B2C "
        "(Business-to-Customer). Determine this based on quantity discussed, buyer "
        "persona (e.g., retailer, distributor vs. individual), and the use case "
        "mentioned in the call."
    ),
    "price": (
        "The exact price information mentioned in the call for a product. Only extract "
        "what is explicitly spoken — this includes the base price, GST/taxes, delivery "
        "charges, and any discounts mentioned. Do not infer or guess prices."
    ),
    "specifications": (
        "All technical or defining characteristics of a product mentioned in the call "
        "(e.g., size, weight, material, brand, model, colour, grade). Extract only "
        "product-defining details that are explicitly stated; do not guess or infer "
        "specification details."
    ),
    "quantity_required": (
        "The amount of product the buyer wants to purchase, if explicitly mentioned "
        "during the call. Include the unit (e.g., kg, pieces, boxes, litres). Do not "
        "guess if not mentioned."
    ),
    "call_purpose": (
        "The main intention or reason of the buyer for calling the seller. Common "
        "purposes include: product inquiry, price negotiation, placing an order, "
        "checking availability, complaint, or follow-up."
    ),
    "all_languages": (
        "List of every language used anywhere in the conversation, including languages "
        "used only briefly or for a single sentence. This must be a complete list."
    ),
    "primary_language": (
        "The main language used for the majority of the conversation. If the "
        "conversation switches mid-way, identify the language that dominates."
    ),
    "product_name": (
        "The name of the product discussed in the call. Prefer the seller's product "
        "name or the most specific product name mentioned. Do not use generic category "
        "names if a specific product name was mentioned."
    ),
    "in_stock": (
        "Indicates whether the seller explicitly confirmed that the product is "
        "available/in stock. Value should be true if availability was confirmed, false "
        "if explicitly denied or not mentioned clearly."
    ),
    "is_buyer_interested": (
        "Indicates whether the buyer showed genuine interest in purchasing the product "
        "during the call. Signals of interest include: asking for a quotation, asking "
        "about delivery, asking about payment terms, or explicitly saying they want to "
        "buy. Value should be True or False."
    ),
    "buyer_next_steps": (
        "What the buyer plans to do next after the call ends. Extract only if "
        "explicitly mentioned by the buyer (e.g., 'I will call back', 'I will share "
        "my address', 'I need to check with my team'). Do not infer."
    ),
    "seller_next_steps": (
        "What the seller says they will do next after the call (e.g., 'I will send "
        "you a quotation', 'I will check and call you back', 'I will arrange delivery'). "
        "Extract only explicitly stated seller commitments."
    ),
}

# Ordered list for consistent display and prompt construction
PARAMETER_ORDER = list(PARAMETER_DEFINITIONS.keys())
