"""Shared test fixtures for the agent-dag-pipeline test suite."""

from __future__ import annotations

import pytest

from agent_dag.shared.data_contracts import ContentBlockOutput, StoreContext


@pytest.fixture
def sample_store_context() -> StoreContext:
    """A valid store context for testing."""
    return StoreContext(
        store_id="test_store_001",
        store_name="Test Store Madrid",
        city="Madrid",
        country_code="ES",
        country_name="Spain",
        language="es",
        language_name="Spanish",
        timezone="Europe/Madrid",
    )


@pytest.fixture
def sample_product() -> dict[str, str]:
    """A valid product dict for testing."""
    return {
        "product-name": "Samsung Galaxy S24 Ultra",
        "brand": "Samsung",
        "sku": "SGS24U-256-BLK",
        "category": "Smartphones",
    }


@pytest.fixture
def clean_content() -> ContentBlockOutput:
    """A content block with no policy violations."""
    return ContentBlockOutput(
        full_description="The Samsung Galaxy S24 Ultra represents the pinnacle of mobile technology.",
        short_description="Premium smartphone with advanced AI features.",
        store_welcome_line="Discover the Samsung Galaxy S24 Ultra at our Madrid store.",
        meta_title="Samsung Galaxy S24 Ultra Madrid",
        meta_description="Explore the Samsung Galaxy S24 Ultra with advanced AI, 200MP camera, and S Pen at our Madrid location.",
        image_alt_tag="Samsung Galaxy S24 Ultra smartphone in titanium black",
        key_features=["200MP camera", "S Pen included", "Titanium frame"],
        faq_items=[{"question": "Does it include S Pen?", "answer": "Yes, the S24 Ultra includes the S Pen built in."}],
    )
