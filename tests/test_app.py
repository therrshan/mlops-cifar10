import pytest
import json
from application import app

# Set up a test client
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Dummy test to verify the root route
def test_root(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Upload Image" in response.data  # Check if the string is in the HTML content
