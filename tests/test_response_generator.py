"""
Tests for Response Generation System.

Phase 2 of Professional Chat Agent implementation.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from workflow_composer.agents.intent.response_generator import (
    ResponseGenerator,
    ResponseTemplate,
    Response,
    ResponseComponent,
    ResponseType,
    ResponseTone,
    Suggestion,
    Card,
    CodeBlock,
    TableData,
    ProgressIndicator,
    TemplateRenderer,
    ResponseHistoryTracker,
    create_response_generator,
)
from workflow_composer.agents.intent.dialog_state_machine import DialogState


class TestResponseTone:
    """Test ResponseTone enum."""
    
    def test_all_tones_defined(self):
        """Test all expected tones exist."""
        assert ResponseTone.PROFESSIONAL.value == "professional"
        assert ResponseTone.FRIENDLY.value == "friendly"
        assert ResponseTone.CONCISE.value == "concise"
        assert ResponseTone.HELPFUL.value == "helpful"
        assert ResponseTone.TECHNICAL.value == "technical"
    
    def test_tone_count(self):
        """Test we have expected number of tones."""
        assert len(ResponseTone) == 5


class TestResponseType:
    """Test ResponseType enum."""
    
    def test_all_types_defined(self):
        """Test all expected types exist."""
        types = [
            ResponseType.TEXT,
            ResponseType.CARD,
            ResponseType.SUGGESTION,
            ResponseType.CONFIRMATION,
            ResponseType.ERROR,
            ResponseType.PROGRESS,
            ResponseType.CODE,
            ResponseType.LIST,
            ResponseType.TABLE,
            ResponseType.WORKFLOW_PREVIEW,
        ]
        assert len(types) == 10


class TestSuggestion:
    """Test Suggestion dataclass."""
    
    def test_basic_suggestion(self):
        """Test creating a basic suggestion."""
        suggestion = Suggestion(text="Click here")
        assert suggestion.text == "Click here"
        assert suggestion.action is None
        assert suggestion.metadata == {}
    
    def test_suggestion_with_action(self):
        """Test suggestion with action."""
        suggestion = Suggestion(text="Run workflow", action="execute:workflow_1")
        assert suggestion.action == "execute:workflow_1"
    
    def test_suggestion_with_metadata(self):
        """Test suggestion with metadata."""
        suggestion = Suggestion(
            text="Select this",
            action="select",
            metadata={"id": 123, "type": "workflow"}
        )
        assert suggestion.metadata["id"] == 123


class TestCard:
    """Test Card dataclass."""
    
    def test_basic_card(self):
        """Test creating a basic card."""
        card = Card(title="Test Card", content="This is content")
        assert card.title == "Test Card"
        assert card.content == "This is content"
        assert card.image_url is None
        assert card.actions == []
    
    def test_card_with_actions(self):
        """Test card with action buttons."""
        card = Card(
            title="Confirm",
            content="Do you want to proceed?",
            actions=[
                Suggestion(text="Yes", action="confirm"),
                Suggestion(text="No", action="cancel")
            ]
        )
        assert len(card.actions) == 2


class TestCodeBlock:
    """Test CodeBlock dataclass."""
    
    def test_basic_code_block(self):
        """Test creating a code block."""
        code = CodeBlock(code="print('hello')", language="python")
        assert code.code == "print('hello')"
        assert code.language == "python"
        assert code.copy_button is True
    
    def test_code_block_with_title(self):
        """Test code block with title."""
        code = CodeBlock(
            code="nextflow run main.nf",
            language="bash",
            title="Run Command"
        )
        assert code.title == "Run Command"


class TestTableData:
    """Test TableData dataclass."""
    
    def test_basic_table(self):
        """Test creating a table."""
        table = TableData(
            headers=["Name", "Status"],
            rows=[["Job1", "Running"], ["Job2", "Complete"]]
        )
        assert len(table.headers) == 2
        assert len(table.rows) == 2
    
    def test_table_with_title(self):
        """Test table with title."""
        table = TableData(
            headers=["Tool", "Version"],
            rows=[["BWA", "0.7.17"]],
            title="Installed Tools"
        )
        assert table.title == "Installed Tools"


class TestProgressIndicator:
    """Test ProgressIndicator dataclass."""
    
    def test_basic_progress(self):
        """Test creating progress indicator."""
        progress = ProgressIndicator(
            current_step=2,
            total_steps=5,
            current_step_name="Alignment",
            percentage=40.0
        )
        assert progress.current_step == 2
        assert progress.total_steps == 5
        assert progress.percentage == 40.0
    
    def test_progress_with_eta(self):
        """Test progress with estimated time."""
        progress = ProgressIndicator(
            current_step=3,
            total_steps=10,
            current_step_name="Processing",
            percentage=30.0,
            estimated_time_remaining=120.5
        )
        assert progress.estimated_time_remaining == 120.5


class TestResponse:
    """Test Response dataclass."""
    
    def test_basic_response(self):
        """Test creating a basic response."""
        response = Response(primary_text="Hello!")
        assert response.primary_text == "Hello!"
        assert response.components == []
        assert response.suggestions == []
        assert response.tone == ResponseTone.PROFESSIONAL
    
    def test_response_with_components(self):
        """Test response with multiple components."""
        response = Response(
            primary_text="Here are your results:",
            components=[
                ResponseComponent(type=ResponseType.TEXT, content="Some text"),
                ResponseComponent(type=ResponseType.LIST, content=["item1", "item2"])
            ]
        )
        assert len(response.components) == 2
    
    def test_response_to_dict(self):
        """Test response serialization."""
        response = Response(
            primary_text="Test",
            suggestions=[Suggestion(text="Next")],
            tone=ResponseTone.FRIENDLY
        )
        
        result = response.to_dict()
        
        assert result["primary_text"] == "Test"
        assert result["tone"] == "friendly"
        assert len(result["suggestions"]) == 1
        assert "timestamp" in result
    
    def test_response_serializes_card(self):
        """Test that cards are properly serialized."""
        card = Card(
            title="Test",
            content="Content",
            actions=[Suggestion(text="Action", action="do")]
        )
        response = Response(
            primary_text="Check this out:",
            components=[ResponseComponent(type=ResponseType.CARD, content=card)]
        )
        
        result = response.to_dict()
        card_content = result["components"][0]["content"]
        
        assert card_content["title"] == "Test"
        assert len(card_content["actions"]) == 1


class TestResponseTemplate:
    """Test ResponseTemplate dataclass."""
    
    def test_basic_template(self):
        """Test creating a basic template."""
        template = ResponseTemplate(
            intent="greeting",
            variations=["Hello!", "Hi there!"]
        )
        assert template.intent == "greeting"
        assert len(template.variations) == 2
    
    def test_template_with_conditions(self):
        """Test template with conditions."""
        template = ResponseTemplate(
            intent="help",
            variations=["I can help with that."],
            conditions={"user_level": "beginner"}
        )
        
        # Test matching conditions
        assert template.matches_conditions({"user_level": "beginner"})
        assert not template.matches_conditions({"user_level": "expert"})
    
    def test_template_condition_with_list(self):
        """Test condition matching with list of values."""
        template = ResponseTemplate(
            intent="greeting",
            variations=["Welcome!"],
            conditions={"language": ["en", "es"]}
        )
        
        assert template.matches_conditions({"language": "en"})
        assert template.matches_conditions({"language": "es"})
        assert not template.matches_conditions({"language": "fr"})
    
    def test_template_condition_with_callable(self):
        """Test condition matching with callable."""
        template = ResponseTemplate(
            intent="response",
            variations=["Large result set."],
            conditions={"result_count": lambda x: x and x > 100}
        )
        
        assert template.matches_conditions({"result_count": 150})
        assert not template.matches_conditions({"result_count": 50})
        assert not template.matches_conditions({})


class TestTemplateRenderer:
    """Test TemplateRenderer class."""
    
    def test_basic_rendering(self):
        """Test basic placeholder substitution."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"
    
    def test_missing_placeholder_default(self):
        """Test missing placeholder with default."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name|Guest}!", {})
        assert result == "Hello Guest!"
    
    def test_missing_placeholder_no_default(self):
        """Test missing placeholder without default."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name}!", {})
        assert result == "Hello [name]!"
    
    def test_filter_upper(self):
        """Test uppercase filter."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name:upper}!", {"name": "world"})
        assert result == "Hello WORLD!"
    
    def test_filter_lower(self):
        """Test lowercase filter."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name:lower}!", {"name": "WORLD"})
        assert result == "Hello world!"
    
    def test_filter_title(self):
        """Test title case filter."""
        renderer = TemplateRenderer()
        result = renderer.render("Hello {name:title}!", {"name": "john doe"})
        assert result == "Hello John Doe!"
    
    def test_multiple_placeholders(self):
        """Test multiple placeholders in one template."""
        renderer = TemplateRenderer()
        result = renderer.render(
            "{greeting} {name}, your {item} is ready.",
            {"greeting": "Hello", "name": "User", "item": "workflow"}
        )
        assert result == "Hello User, your workflow is ready."
    
    def test_custom_filter(self):
        """Test adding custom filter."""
        renderer = TemplateRenderer()
        renderer.add_filter("reverse", lambda x: str(x)[::-1])
        
        result = renderer.render("Message: {text:reverse}", {"text": "hello"})
        assert result == "Message: olleh"


class TestResponseHistoryTracker:
    """Test ResponseHistoryTracker class."""
    
    def test_record_response(self):
        """Test recording a response."""
        tracker = ResponseHistoryTracker()
        tracker.record_response("Hello!")
        
        assert len(tracker.history) == 1
        assert tracker.history[0][0] == "Hello!"
    
    def test_was_recently_used(self):
        """Test detecting recently used responses."""
        tracker = ResponseHistoryTracker(repetition_window=3)
        
        tracker.record_response("Response 1")
        tracker.record_response("Response 2")
        
        assert tracker.was_recently_used("Response 1")
        assert tracker.was_recently_used("Response 2")
        assert not tracker.was_recently_used("Response 3")
    
    def test_repetition_window(self):
        """Test that old responses fall out of window."""
        tracker = ResponseHistoryTracker(repetition_window=2)
        
        tracker.record_response("Response 1")
        tracker.record_response("Response 2")
        tracker.record_response("Response 3")
        
        # Response 1 should be outside the window now
        assert not tracker.was_recently_used("Response 1")
        assert tracker.was_recently_used("Response 2")
        assert tracker.was_recently_used("Response 3")
    
    def test_max_history(self):
        """Test history trimming at max size."""
        tracker = ResponseHistoryTracker(max_history=5)
        
        for i in range(10):
            tracker.record_response(f"Response {i}")
        
        assert len(tracker.history) == 5
        # Only last 5 should remain
        assert tracker.history[0][0] == "Response 5"
    
    def test_template_usage_tracking(self):
        """Test tracking template usage."""
        tracker = ResponseHistoryTracker()
        
        tracker.record_response("Hello!", "greeting:v1")
        tracker.record_response("Hi there!", "greeting:v2")
        tracker.record_response("Hello!", "greeting:v1")
        
        assert tracker.get_template_usage_count("greeting:v1") == 2
        assert tracker.get_template_usage_count("greeting:v2") == 1


class TestResponseGenerator:
    """Test ResponseGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = ResponseGenerator()
        
        assert generator.default_tone == ResponseTone.PROFESSIONAL
        assert generator.history is not None
        assert len(generator.templates) > 0  # Has default templates
    
    def test_initialization_without_history(self):
        """Test generator without history tracking."""
        generator = ResponseGenerator(enable_history_tracking=False)
        assert generator.history is None
    
    def test_default_templates_loaded(self):
        """Test that default templates are loaded."""
        generator = ResponseGenerator()
        
        # Should have common intents
        assert "greeting" in generator.templates
        assert "create_workflow" in generator.templates
        assert "help" in generator.templates
        assert "error" in generator.templates
    
    def test_add_template(self):
        """Test adding a custom template."""
        generator = ResponseGenerator()
        
        template = ResponseTemplate(
            intent="custom_intent",
            variations=["Custom response 1", "Custom response 2"]
        )
        generator.add_template(template)
        
        assert "custom_intent" in generator.templates
        assert len(generator.templates["custom_intent"]) == 1
    
    def test_generate_greeting(self):
        """Test generating a greeting response."""
        generator = ResponseGenerator()
        response = generator.generate("greeting")
        
        assert response.primary_text  # Should have text
        assert len(response.suggestions) > 0  # Greeting has suggestions
        assert response.metadata["intent"] == "greeting"
    
    def test_generate_with_variables(self):
        """Test generating response with variables."""
        generator = ResponseGenerator()
        response = generator.generate(
            "create_workflow",
            variables={"workflow_type": "RNA-seq"}
        )
        
        assert "RNA-seq" in response.primary_text
    
    def test_generate_fallback(self):
        """Test fallback for unknown intent."""
        generator = ResponseGenerator()
        response = generator.generate("nonexistent_intent")
        
        assert response.primary_text
        assert response.metadata.get("template_used") is False
    
    def test_generate_state_response(self):
        """Test generating state-specific responses."""
        generator = ResponseGenerator()
        
        response = generator.generate_state_response(DialogState.IDLE)
        assert response.primary_text
        assert response.metadata["state"] == "idle"
    
    def test_generate_state_response_timeout(self):
        """Test generating timeout response for state."""
        generator = ResponseGenerator()
        
        response = generator.generate_state_response(
            DialogState.IDLE,
            response_type="timeout"
        )
        assert response.primary_text
    
    def test_generate_slot_prompt(self):
        """Test generating slot prompts."""
        generator = ResponseGenerator()
        
        response = generator.generate_slot_prompt(
            slot_name="organism",
            slot_description="The target organism (e.g., human, mouse)",
            examples=["human", "mouse", "zebrafish"]
        )
        
        assert "organism" in response.primary_text
        assert len(response.suggestions) == 3
        assert response.metadata["slot_name"] == "organism"
    
    def test_generate_disambiguation(self):
        """Test generating disambiguation responses."""
        generator = ResponseGenerator()
        
        options = [
            {"label": "RNA-seq", "description": "Gene expression analysis"},
            {"label": "ChIP-seq", "description": "Protein-DNA binding"},
            {"label": "ATAC-seq", "description": "Chromatin accessibility"}
        ]
        
        response = generator.generate_disambiguation("seq analysis", options)
        
        assert "seq analysis" in response.primary_text
        assert len(response.suggestions) == 3
        assert response.metadata["options_count"] == 3
    
    def test_generate_confirmation(self):
        """Test generating confirmation responses."""
        generator = ResponseGenerator()
        
        response = generator.generate_confirmation(
            "run the RNA-seq pipeline",
            details={"organism": "human", "samples": 6}
        )
        
        assert "run the RNA-seq pipeline" in response.primary_text
        assert len(response.components) == 1  # Should have details card
        assert response.components[0].type == ResponseType.CARD
    
    def test_generate_error(self):
        """Test generating error responses."""
        generator = ResponseGenerator()
        
        response = generator.generate_error(
            "File not found",
            recovery_suggestion="Check the file path and try again.",
            error_code="FILE_404"
        )
        
        assert "File not found" in response.primary_text
        assert response.metadata["error_code"] == "FILE_404"
        assert len(response.suggestions) >= 2  # Retry, help options
    
    def test_generate_progress(self):
        """Test generating progress responses."""
        generator = ResponseGenerator()
        
        response = generator.generate_progress(
            current_step=3,
            total_steps=10,
            current_step_name="Alignment",
            estimated_time=300.0
        )
        
        assert len(response.components) == 1
        assert response.components[0].type == ResponseType.PROGRESS
        
        progress = response.components[0].content
        assert progress.current_step == 3
        assert progress.percentage == 30.0
    
    def test_generate_workflow_preview(self):
        """Test generating workflow preview."""
        generator = ResponseGenerator()
        
        steps = [
            {"name": "QC", "tool": "FastQC", "description": "Quality control"},
            {"name": "Trim", "tool": "Trimmomatic", "description": "Adapter trimming"},
            {"name": "Align", "tool": "STAR", "description": "Read alignment"}
        ]
        params = {"genome": "hg38", "threads": 8}
        
        response = generator.generate_workflow_preview(
            "RNA-seq",
            steps=steps,
            parameters=params
        )
        
        assert "RNA-seq" in response.primary_text
        assert len(response.components) == 2  # Table + card
        
        table_component = response.components[0]
        assert table_component.type == ResponseType.TABLE
    
    def test_generate_code_response(self):
        """Test generating code response."""
        generator = ResponseGenerator()
        
        code = """
process ALIGN {
    input:
    path reads
    
    output:
    path "*.bam"
    
    script:
    '''
    bwa mem ref.fa ${reads} | samtools sort -o out.bam
    '''
}
"""
        response = generator.generate_code_response(
            code=code,
            language="nextflow",
            title="Alignment Process",
            explanation="Here's the alignment process definition:"
        )
        
        assert response.primary_text == "Here's the alignment process definition:"
        assert len(response.components) == 1
        assert response.components[0].type == ResponseType.CODE
        
        code_block = response.components[0].content
        assert code_block.language == "nextflow"
        assert code_block.title == "Alignment Process"
    
    def test_generate_list_response(self):
        """Test generating list response."""
        generator = ResponseGenerator()
        
        response = generator.generate_list_response(
            title="Available workflows:",
            items=["RNA-seq", "ChIP-seq", "ATAC-seq"],
            item_actions={"RNA-seq": "select:rna", "ChIP-seq": "select:chip"}
        )
        
        assert "Available workflows" in response.primary_text
        assert len(response.components) == 1
        assert response.components[0].type == ResponseType.LIST
        assert len(response.suggestions) == 2
    
    def test_set_tone(self):
        """Test changing default tone."""
        generator = ResponseGenerator()
        generator.set_tone(ResponseTone.FRIENDLY)
        
        assert generator.default_tone == ResponseTone.FRIENDLY
    
    def test_add_custom_template(self):
        """Test adding custom template via method."""
        generator = ResponseGenerator()
        
        generator.add_custom_template(
            intent="my_intent",
            variations=["Response A", "Response B"],
            follow_up_suggestions=["Next step"],
            priority=10
        )
        
        response = generator.generate("my_intent")
        assert response.primary_text in ["Response A", "Response B"]
    
    def test_load_templates_from_config(self):
        """Test loading templates from config dict."""
        generator = ResponseGenerator()
        
        config = {
            "templates": [
                {
                    "intent": "config_intent",
                    "variations": ["Config response 1", "Config response 2"],
                    "tone": "friendly",
                    "follow_up_suggestions": ["Do more"]
                }
            ]
        }
        
        generator.load_templates_from_config(config)
        
        assert "config_intent" in generator.templates
        response = generator.generate("config_intent")
        assert response.tone == ResponseTone.FRIENDLY
    
    def test_get_statistics(self):
        """Test getting generator statistics."""
        generator = ResponseGenerator()
        
        # Generate some responses
        generator.generate("greeting")
        generator.generate("help")
        
        stats = generator.get_statistics()
        
        assert "total_templates" in stats
        assert "intents_covered" in stats
        assert "total_responses_generated" in stats
        assert stats["total_responses_generated"] >= 2
    
    def test_avoids_repetition(self):
        """Test that generator tries to avoid repetition."""
        generator = ResponseGenerator()
        
        # Add a template with many variations
        generator.add_custom_template(
            intent="varied_response",
            variations=[f"Response variant {i}" for i in range(10)]
        )
        
        # Generate multiple responses
        responses = [generator.generate("varied_response").primary_text for _ in range(5)]
        
        # Should have variety (not all the same)
        unique_responses = set(responses)
        assert len(unique_responses) > 1
    
    def test_conditional_template_selection(self):
        """Test selecting templates based on conditions."""
        generator = ResponseGenerator()
        
        # Add low priority template without conditions
        generator.add_custom_template(
            intent="conditional_test",
            variations=["Default response"],
            priority=0
        )
        
        # Add high priority template with conditions
        generator.add_template(ResponseTemplate(
            intent="conditional_test",
            variations=["Expert response"],
            conditions={"user_level": "expert"},
            priority=10
        ))
        
        # Test without condition match
        response1 = generator.generate("conditional_test", context={})
        
        # Test with condition match
        response2 = generator.generate(
            "conditional_test",
            context={"user_level": "expert"}
        )
        
        assert "Expert" in response2.primary_text


class TestCreateResponseGenerator:
    """Test create_response_generator factory function."""
    
    def test_basic_creation(self):
        """Test basic factory creation."""
        generator = create_response_generator()
        assert isinstance(generator, ResponseGenerator)
        assert generator.default_tone == ResponseTone.PROFESSIONAL
    
    def test_create_with_tone(self):
        """Test creating with specific tone."""
        generator = create_response_generator(tone=ResponseTone.FRIENDLY)
        assert generator.default_tone == ResponseTone.FRIENDLY
    
    def test_create_with_custom_templates(self):
        """Test creating with custom templates."""
        custom = [
            ResponseTemplate(
                intent="custom",
                variations=["Custom!"]
            )
        ]
        
        generator = create_response_generator(custom_templates=custom)
        
        response = generator.generate("custom")
        assert response.primary_text == "Custom!"


class TestIntegration:
    """Integration tests for Response Generator with Dialog State Machine."""
    
    def test_response_for_each_state(self):
        """Test that we can generate responses for all dialog states."""
        generator = ResponseGenerator()
        
        for state in DialogState:
            response = generator.generate_state_response(state)
            assert response.primary_text
            assert response.metadata.get("state") == state.name.lower()
    
    def test_response_serialization_roundtrip(self):
        """Test that responses can be serialized and contain expected data."""
        generator = ResponseGenerator()
        
        response = generator.generate_confirmation(
            "execute workflow",
            details={"type": "RNA-seq", "samples": 3}
        )
        
        data = response.to_dict()
        
        assert isinstance(data, dict)
        assert "primary_text" in data
        assert "components" in data
        assert isinstance(data["timestamp"], str)  # ISO format
    
    def test_complex_workflow_response(self):
        """Test generating a complex workflow-related response."""
        generator = ResponseGenerator()
        
        # Simulate a complete workflow presentation
        response = generator.generate_workflow_preview(
            "RNA-seq Differential Expression",
            steps=[
                {"name": "FastQC", "tool": "fastqc", "description": "Quality assessment"},
                {"name": "Trim", "tool": "trim_galore", "description": "Adapter trimming"},
                {"name": "Align", "tool": "STAR", "description": "Genome alignment"},
                {"name": "Count", "tool": "featureCounts", "description": "Read counting"},
                {"name": "DE", "tool": "DESeq2", "description": "Differential expression"}
            ],
            parameters={
                "genome": "GRCh38",
                "annotation": "Ensembl v104",
                "threads": 16,
                "comparison": "treatment vs control"
            }
        )
        
        # Verify structure
        assert len(response.components) == 2
        
        # Check table
        table = response.components[0].content
        assert len(table.rows) == 5
        
        # Check card with parameters
        card = response.components[1].content
        assert "genome" in card.content.lower() or "GRCh38" in card.content


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_variables(self):
        """Test with empty variables dict."""
        generator = ResponseGenerator()
        response = generator.generate("greeting", variables={})
        assert response.primary_text
    
    def test_none_variables(self):
        """Test with None variables."""
        generator = ResponseGenerator()
        response = generator.generate("greeting", variables=None)
        assert response.primary_text
    
    def test_empty_suggestions(self):
        """Test generating response when suggestions are disabled."""
        generator = ResponseGenerator()
        response = generator.generate("greeting", include_suggestions=False)
        assert response.suggestions == []
    
    def test_template_with_missing_placeholders(self):
        """Test template when some placeholders are missing values."""
        generator = ResponseGenerator()
        
        generator.add_custom_template(
            intent="test_missing",
            variations=["Hello {name}, your {item} is ready."]
        )
        
        # Only provide some variables
        response = generator.generate(
            "test_missing",
            variables={"name": "User"}
        )
        
        # Should handle gracefully
        assert "User" in response.primary_text
        assert "[item]" in response.primary_text  # Default placeholder
    
    def test_progress_zero_total(self):
        """Test progress with zero total steps."""
        generator = ResponseGenerator()
        
        response = generator.generate_progress(
            current_step=0,
            total_steps=0,
            current_step_name="Initializing"
        )
        
        progress = response.components[0].content
        assert progress.percentage == 0
    
    def test_empty_disambiguation_options(self):
        """Test disambiguation with empty options."""
        generator = ResponseGenerator()
        
        response = generator.generate_disambiguation("query", [])
        
        assert response.suggestions == []
        assert response.metadata["options_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
