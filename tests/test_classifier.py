"""Tests for OpenAlex Topic Classifier."""

import pytest
from openalex_classifier import TopicClassifier, Config


class TestTopicClassifier:
    """Test the main classifier functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create and initialize classifier."""
        clf = TopicClassifier()
        clf.initialize()
        return clf
    
    def test_classify_single_record(self, classifier):
        """Test classifying a single record."""
        record = {
            "title": "Climate change impact on marine ecosystems",
            "subjects": ["Climate", "Marine Biology"]
        }
        
        result = classifier.classify(record)
        
        assert 'topic' in result
        assert 'subfield' in result
        assert 'field' in result
        assert 'domain' in result
        
        if result['topic']:
            assert 'id' in result['topic']
            assert 'name' in result['topic']
            assert 'score' in result['topic']
            assert 0 <= result['topic']['score'] <= 1
    
    def test_classify_batch(self, classifier):
        """Test batch classification."""
        records = [
            {"title": "Machine learning for drug discovery"},
            {"title": "Archaeological excavation methods"},
            {"title": "Galaxy formation and evolution"},
        ]
        
        results = classifier.classify_batch(records)
        
        assert len(results) == 3
        for result in results:
            assert 'dataset_id' in result
    
    def test_empty_batch(self, classifier):
        """Test with empty batch."""
        results = classifier.classify_batch([])
        assert results == []
    
    def test_missing_title(self, classifier):
        """Test record with missing title."""
        record = {"subjects": ["Physics"]}
        result = classifier.classify(record)
        
        # Should still return a result (may have low score)
        assert 'topic' in result


class TestConfig:
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.min_score == 0.40
        assert config.batch_size == 256
        assert config.pca_dims == 256
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(min_score=0.60, batch_size=128)
        
        assert config.min_score == 0.60
        assert config.batch_size == 128

