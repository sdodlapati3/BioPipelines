"""
Tests for Rich Response Helpers.

Phase 6 of Professional Chat Agent implementation.
"""

import pytest
from workflow_composer.agents.intent.rich_responses import (
    MessageFormatter,
    MessageBuilder,
    MessageFormat,
    ComponentType,
    CalloutType,
    ButtonStyle,
    ListStyle,
    RichComponent,
    TextComponent,
    HeadingComponent,
    ListComponent,
    CodeComponent,
    TableComponent,
    ImageComponent,
    LinkComponent,
    ButtonComponent,
    CardComponent,
    CalloutComponent,
    ProgressComponent,
    QuoteComponent,
    CarouselComponent,
    FormComponent,
    InputComponent,
    PlainTextAdapter,
    MarkdownAdapter,
    HTMLAdapter,
    SlackAdapter,
    strip_formatting,
    truncate_text,
    word_wrap,
    format_duration,
    format_number,
    format_bytes,
    get_message_formatter,
    reset_message_formatter,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_message_format(self):
        """Test MessageFormat enum."""
        assert MessageFormat.PLAIN_TEXT.value == "plain_text"
        assert MessageFormat.MARKDOWN.value == "markdown"
        assert MessageFormat.HTML.value == "html"
        assert MessageFormat.SLACK.value == "slack"
    
    def test_component_type(self):
        """Test ComponentType enum."""
        types = [
            ComponentType.TEXT, ComponentType.HEADING, ComponentType.LIST,
            ComponentType.CODE, ComponentType.TABLE, ComponentType.IMAGE,
            ComponentType.BUTTON, ComponentType.CARD, ComponentType.CALLOUT,
        ]
        assert len(types) == 9
    
    def test_callout_type(self):
        """Test CalloutType enum."""
        types = [
            CalloutType.INFO, CalloutType.SUCCESS, CalloutType.WARNING,
            CalloutType.ERROR, CalloutType.TIP, CalloutType.NOTE,
        ]
        assert len(types) == 6
    
    def test_button_style(self):
        """Test ButtonStyle enum."""
        assert ButtonStyle.PRIMARY.value == "primary"
        assert ButtonStyle.DANGER.value == "danger"
    
    def test_list_style(self):
        """Test ListStyle enum."""
        assert ListStyle.BULLET.value == "bullet"
        assert ListStyle.NUMBERED.value == "numbered"
        assert ListStyle.CHECKLIST.value == "checklist"


class TestTextComponent:
    """Test TextComponent class."""
    
    def test_create_basic(self):
        """Test creating basic text component."""
        component = TextComponent(content="Hello")
        assert component.content == "Hello"
        assert component.type == ComponentType.TEXT
    
    def test_formatting_options(self):
        """Test formatting options."""
        component = TextComponent(
            content="Text",
            bold=True,
            italic=True,
            strikethrough=True,
            code=True
        )
        assert component.bold
        assert component.italic
        assert component.strikethrough
        assert component.code
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        component = TextComponent(content="Test", bold=True)
        d = component.to_dict()
        
        assert d["content"] == "Test"
        assert d["bold"] == True


class TestHeadingComponent:
    """Test HeadingComponent class."""
    
    def test_create_heading(self):
        """Test creating heading."""
        component = HeadingComponent(content="Title", level=1)
        assert component.content == "Title"
        assert component.level == 1
    
    def test_default_level(self):
        """Test default heading level."""
        component = HeadingComponent(content="Title")
        assert component.level == 1


class TestListComponent:
    """Test ListComponent class."""
    
    def test_create_bullet_list(self):
        """Test creating bullet list."""
        component = ListComponent(items=["a", "b", "c"])
        assert len(component.items) == 3
        assert component.style == ListStyle.BULLET
    
    def test_create_numbered_list(self):
        """Test creating numbered list."""
        component = ListComponent(items=["1", "2"], style=ListStyle.NUMBERED)
        assert component.style == ListStyle.NUMBERED
    
    def test_checklist(self):
        """Test checklist with checked items."""
        component = ListComponent(
            items=["Task 1", "Task 2"],
            style=ListStyle.CHECKLIST,
            checked={0: True, 1: False}
        )
        assert component.checked[0] == True
        assert component.checked[1] == False


class TestCodeComponent:
    """Test CodeComponent class."""
    
    def test_create_code_block(self):
        """Test creating code block."""
        component = CodeComponent(
            content="print('hello')",
            language="python",
            filename="test.py"
        )
        assert "print" in component.content
        assert component.language == "python"
        assert component.filename == "test.py"
    
    def test_line_numbers(self):
        """Test line numbers option."""
        component = CodeComponent(
            content="code",
            line_numbers=True,
            highlight_lines=[1, 3]
        )
        assert component.line_numbers
        assert 1 in component.highlight_lines


class TestTableComponent:
    """Test TableComponent class."""
    
    def test_create_table(self):
        """Test creating table."""
        component = TableComponent(
            headers=["Name", "Value"],
            rows=[["A", "1"], ["B", "2"]]
        )
        assert len(component.headers) == 2
        assert len(component.rows) == 2
    
    def test_table_with_caption(self):
        """Test table with caption."""
        component = TableComponent(
            headers=["Col"],
            rows=[],
            caption="Test Table"
        )
        assert component.caption == "Test Table"


class TestImageComponent:
    """Test ImageComponent class."""
    
    def test_create_image(self):
        """Test creating image."""
        component = ImageComponent(
            url="http://example.com/image.png",
            alt_text="Example",
            caption="An image"
        )
        assert "example.com" in component.url
        assert component.alt_text == "Example"
    
    def test_dimensions(self):
        """Test image dimensions."""
        component = ImageComponent(
            url="http://example.com/img.png",
            width=100,
            height=200
        )
        assert component.width == 100
        assert component.height == 200


class TestButtonComponent:
    """Test ButtonComponent class."""
    
    def test_create_button(self):
        """Test creating button."""
        component = ButtonComponent(
            text="Click me",
            action="submit",
            style=ButtonStyle.PRIMARY
        )
        assert component.text == "Click me"
        assert component.action == "submit"
        assert component.style == ButtonStyle.PRIMARY
    
    def test_button_data(self):
        """Test button with data."""
        component = ButtonComponent(
            text="Delete",
            action="delete",
            data={"item_id": 123}
        )
        assert component.data["item_id"] == 123


class TestCardComponent:
    """Test CardComponent class."""
    
    def test_create_card(self):
        """Test creating card."""
        component = CardComponent(
            title="Card Title",
            subtitle="Subtitle",
            body="Card content"
        )
        assert component.title == "Card Title"
        assert component.subtitle == "Subtitle"
        assert component.body == "Card content"
    
    def test_card_with_actions(self):
        """Test card with action buttons."""
        actions = [
            ButtonComponent(text="Action 1", action="a1"),
            ButtonComponent(text="Action 2", action="a2"),
        ]
        component = CardComponent(
            title="Title",
            actions=actions
        )
        assert len(component.actions) == 2


class TestCalloutComponent:
    """Test CalloutComponent class."""
    
    def test_create_callout(self):
        """Test creating callout."""
        component = CalloutComponent(
            content="Important message",
            callout_type=CalloutType.WARNING,
            title="Warning"
        )
        assert component.content == "Important message"
        assert component.callout_type == CalloutType.WARNING
    
    def test_callout_types(self):
        """Test different callout types."""
        for callout_type in CalloutType:
            component = CalloutComponent(
                content="Test",
                callout_type=callout_type
            )
            assert component.callout_type == callout_type


class TestProgressComponent:
    """Test ProgressComponent class."""
    
    def test_create_progress(self):
        """Test creating progress indicator."""
        component = ProgressComponent(
            value=75,
            max_value=100,
            label="Loading"
        )
        assert component.value == 75
        assert component.max_value == 100
        assert component.label == "Loading"
    
    def test_animated_progress(self):
        """Test animated progress."""
        component = ProgressComponent(value=50, animated=True)
        assert component.animated


class TestQuoteComponent:
    """Test QuoteComponent class."""
    
    def test_create_quote(self):
        """Test creating quote."""
        component = QuoteComponent(
            content="To be or not to be",
            author="Shakespeare",
            source="Hamlet"
        )
        assert "To be" in component.content
        assert component.author == "Shakespeare"


class TestMessageBuilder:
    """Test MessageBuilder class."""
    
    def test_builder_chain(self):
        """Test fluent builder interface."""
        builder = MessageBuilder()
        result = (builder
            .heading("Title")
            .text("Content")
            .list(["a", "b"])
            .build())
        
        assert len(result) == 3
    
    def test_text_formatting(self):
        """Test text with formatting."""
        builder = MessageBuilder()
        result = builder.text("Bold", bold=True).build()
        
        assert isinstance(result[0], TextComponent)
        assert result[0].bold
    
    def test_code_block(self):
        """Test adding code block."""
        builder = MessageBuilder()
        result = builder.code("print(1)", language="python").build()
        
        assert isinstance(result[0], CodeComponent)
        assert result[0].language == "python"
    
    def test_table(self):
        """Test adding table."""
        builder = MessageBuilder()
        result = builder.table(
            headers=["A", "B"],
            rows=[["1", "2"]]
        ).build()
        
        assert isinstance(result[0], TableComponent)
    
    def test_callout(self):
        """Test adding callout."""
        builder = MessageBuilder()
        result = builder.callout(
            "Warning message",
            CalloutType.WARNING
        ).build()
        
        assert isinstance(result[0], CalloutComponent)
        assert result[0].callout_type == CalloutType.WARNING
    
    def test_divider(self):
        """Test adding divider."""
        builder = MessageBuilder()
        result = builder.divider().build()
        
        assert result[0].type == ComponentType.DIVIDER
    
    def test_clear(self):
        """Test clearing builder."""
        builder = MessageBuilder()
        builder.text("Test").text("More")
        builder.clear()
        result = builder.build()
        
        assert len(result) == 0
    
    def test_add_custom_component(self):
        """Test adding custom component."""
        builder = MessageBuilder()
        custom = RichComponent(type=ComponentType.TEXT, content="Custom")
        result = builder.add(custom).build()
        
        assert len(result) == 1


class TestPlainTextAdapter:
    """Test PlainTextAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return PlainTextAdapter()
    
    def test_render_text(self, adapter):
        """Test rendering text."""
        component = TextComponent(content="Hello")
        result = adapter.render([component])
        assert "Hello" in result
    
    def test_render_heading(self, adapter):
        """Test rendering heading."""
        component = HeadingComponent(content="Title", level=1)
        result = adapter.render([component])
        assert "TITLE" in result
        assert "=" in result
    
    def test_render_list(self, adapter):
        """Test rendering list."""
        component = ListComponent(items=["one", "two"])
        result = adapter.render([component])
        assert "• one" in result
        assert "• two" in result
    
    def test_render_numbered_list(self, adapter):
        """Test rendering numbered list."""
        component = ListComponent(items=["a", "b"], style=ListStyle.NUMBERED)
        result = adapter.render([component])
        assert "1. a" in result
        assert "2. b" in result
    
    def test_render_code(self, adapter):
        """Test rendering code block."""
        component = CodeComponent(content="code")
        result = adapter.render([component])
        assert "code" in result
        assert "---CODE---" in result
    
    def test_render_table(self, adapter):
        """Test rendering table."""
        component = TableComponent(
            headers=["A", "B"],
            rows=[["1", "2"]]
        )
        result = adapter.render([component])
        assert "A" in result
        assert "1" in result
    
    def test_render_progress(self, adapter):
        """Test rendering progress."""
        component = ProgressComponent(value=50, label="Loading")
        result = adapter.render([component])
        assert "Loading" in result
        assert "█" in result
    
    def test_render_callout(self, adapter):
        """Test rendering callout."""
        component = CalloutComponent(
            content="Warning",
            callout_type=CalloutType.WARNING
        )
        result = adapter.render([component])
        assert "⚠" in result
        assert "Warning" in result


class TestMarkdownAdapter:
    """Test MarkdownAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return MarkdownAdapter()
    
    def test_render_bold(self, adapter):
        """Test rendering bold text."""
        component = TextComponent(content="bold", bold=True)
        result = adapter.render([component])
        assert "**bold**" in result
    
    def test_render_italic(self, adapter):
        """Test rendering italic text."""
        component = TextComponent(content="italic", italic=True)
        result = adapter.render([component])
        assert "*italic*" in result
    
    def test_render_heading(self, adapter):
        """Test rendering heading."""
        component = HeadingComponent(content="Title", level=2)
        result = adapter.render([component])
        assert "## Title" in result
    
    def test_render_code(self, adapter):
        """Test rendering code block."""
        component = CodeComponent(content="print(1)", language="python")
        result = adapter.render([component])
        assert "```python" in result
        assert "print(1)" in result
        assert "```" in result
    
    def test_render_table(self, adapter):
        """Test rendering table."""
        component = TableComponent(
            headers=["A", "B"],
            rows=[["1", "2"]]
        )
        result = adapter.render([component])
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "---" in result
    
    def test_render_link(self, adapter):
        """Test rendering link."""
        component = LinkComponent(url="http://example.com", text="Example")
        result = adapter.render([component])
        assert "[Example](http://example.com)" in result
    
    def test_render_image(self, adapter):
        """Test rendering image."""
        component = ImageComponent(
            url="http://example.com/img.png",
            alt_text="Image"
        )
        result = adapter.render([component])
        assert "![Image](http://example.com/img.png)" in result
    
    def test_render_divider(self, adapter):
        """Test rendering divider."""
        component = RichComponent(type=ComponentType.DIVIDER)
        result = adapter.render([component])
        assert "---" in result


class TestHTMLAdapter:
    """Test HTMLAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return HTMLAdapter()
    
    def test_render_text(self, adapter):
        """Test rendering text."""
        component = TextComponent(content="Hello")
        result = adapter.render([component])
        assert "<span>Hello</span>" in result
    
    def test_render_bold(self, adapter):
        """Test rendering bold text."""
        component = TextComponent(content="bold", bold=True)
        result = adapter.render([component])
        assert "<strong>bold</strong>" in result
    
    def test_render_heading(self, adapter):
        """Test rendering heading."""
        component = HeadingComponent(content="Title", level=2)
        result = adapter.render([component])
        assert "<h2>Title</h2>" in result
    
    def test_render_list(self, adapter):
        """Test rendering list."""
        component = ListComponent(items=["a", "b"])
        result = adapter.render([component])
        assert "<ul>" in result
        assert "<li>a</li>" in result
    
    def test_render_table(self, adapter):
        """Test rendering table."""
        component = TableComponent(
            headers=["Col"],
            rows=[["val"]]
        )
        result = adapter.render([component])
        assert "<table>" in result
        assert "<th>Col</th>" in result
        assert "<td>val</td>" in result
    
    def test_escape_html(self, adapter):
        """Test HTML escaping."""
        component = TextComponent(content="<script>alert('xss')</script>")
        result = adapter.render([component])
        assert "&lt;script&gt;" in result
    
    def test_render_button(self, adapter):
        """Test rendering button."""
        component = ButtonComponent(
            text="Click",
            action="submit",
            style=ButtonStyle.PRIMARY
        )
        result = adapter.render([component])
        assert "<button" in result
        assert "btn-primary" in result
        assert "data-action" in result


class TestSlackAdapter:
    """Test SlackAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return SlackAdapter()
    
    def test_render_bold(self, adapter):
        """Test rendering bold text."""
        component = TextComponent(content="bold", bold=True)
        result = adapter.render([component])
        assert "*bold*" in result
    
    def test_render_link(self, adapter):
        """Test rendering link."""
        component = LinkComponent(url="http://example.com", text="Example")
        result = adapter.render([component])
        assert "<http://example.com|Example>" in result
    
    def test_to_blocks(self, adapter):
        """Test converting to Slack blocks."""
        components = [
            HeadingComponent(content="Title"),
            RichComponent(type=ComponentType.DIVIDER),
        ]
        blocks = adapter.to_blocks(components)
        
        assert any(b["type"] == "header" for b in blocks)
        assert any(b["type"] == "divider" for b in blocks)


class TestMessageFormatter:
    """Test MessageFormatter class."""
    
    @pytest.fixture
    def formatter(self):
        """Create formatter instance."""
        return MessageFormatter()
    
    def test_render_markdown(self, formatter):
        """Test rendering to Markdown."""
        components = [TextComponent(content="Hello", bold=True)]
        result = formatter.render(components, MessageFormat.MARKDOWN)
        assert "**Hello**" in result
    
    def test_render_plain_text(self, formatter):
        """Test rendering to plain text."""
        components = [TextComponent(content="Hello")]
        result = formatter.render(components, MessageFormat.PLAIN_TEXT)
        assert "Hello" in result
    
    def test_format_text_convenience(self, formatter):
        """Test format_text convenience method."""
        result = formatter.format_text("Bold", bold=True)
        assert "**Bold**" in result
    
    def test_format_list_convenience(self, formatter):
        """Test format_list convenience method."""
        result = formatter.format_list(["a", "b"])
        assert "- a" in result
        assert "- b" in result
    
    def test_format_code_convenience(self, formatter):
        """Test format_code convenience method."""
        result = formatter.format_code("print(1)", "python")
        assert "```python" in result
    
    def test_format_error(self, formatter):
        """Test format_error convenience method."""
        result = formatter.format_error("Something went wrong")
        assert "Error" in result or "❌" in result
    
    def test_format_success(self, formatter):
        """Test format_success convenience method."""
        result = formatter.format_success("Done!")
        assert "Success" in result or "✅" in result
    
    def test_format_progress(self, formatter):
        """Test format_progress convenience method."""
        result = formatter.format_progress(50, 100, "Loading")
        assert "50" in result
    
    def test_builder_method(self, formatter):
        """Test builder method."""
        builder = formatter.builder()
        assert isinstance(builder, MessageBuilder)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_strip_formatting(self):
        """Test strip_formatting function."""
        text = "**bold** and *italic* with `code`"
        result = strip_formatting(text)
        assert "bold" in result
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
    
    def test_strip_formatting_links(self):
        """Test strip_formatting removes links."""
        text = "[click here](http://example.com)"
        result = strip_formatting(text)
        assert result == "click here"
    
    def test_truncate_text(self):
        """Test truncate_text function."""
        text = "This is a long text that needs truncation"
        result = truncate_text(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncate_text with short text."""
        text = "Short"
        result = truncate_text(text, 20)
        assert result == "Short"
    
    def test_word_wrap(self):
        """Test word_wrap function."""
        text = "This is a long line that needs to be wrapped to a smaller width"
        result = word_wrap(text, width=20)
        lines = result.split("\n")
        assert all(len(line) <= 20 for line in lines)
    
    def test_format_duration_seconds(self):
        """Test format_duration for seconds."""
        result = format_duration(30)
        assert "seconds" in result.lower()
    
    def test_format_duration_minutes(self):
        """Test format_duration for minutes."""
        result = format_duration(120)
        assert "minute" in result.lower()
    
    def test_format_duration_hours(self):
        """Test format_duration for hours."""
        result = format_duration(7200)
        assert "hour" in result.lower()
    
    def test_format_duration_days(self):
        """Test format_duration for days."""
        result = format_duration(172800)
        assert "day" in result.lower()
    
    def test_format_number(self):
        """Test format_number function."""
        assert "1.00K" in format_number(1000)
        assert "1.50M" in format_number(1500000)
    
    def test_format_number_small(self):
        """Test format_number for small numbers."""
        assert "42.00" in format_number(42)
    
    def test_format_bytes(self):
        """Test format_bytes function."""
        assert "KB" in format_bytes(1024)
        assert "MB" in format_bytes(1024 * 1024)
        assert "GB" in format_bytes(1024 * 1024 * 1024)


class TestSingleton:
    """Test singleton functions."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        reset_message_formatter()
    
    def test_get_message_formatter(self):
        """Test getting singleton formatter."""
        f1 = get_message_formatter()
        f2 = get_message_formatter()
        assert f1 is f2
    
    def test_reset_message_formatter(self):
        """Test resetting singleton."""
        f1 = get_message_formatter()
        reset_message_formatter()
        f2 = get_message_formatter()
        assert f1 is not f2


class TestIntegration:
    """Integration tests for rich responses."""
    
    def test_complex_message_building(self):
        """Test building a complex message."""
        message = (MessageBuilder()
            .heading("Welcome to the System", level=1)
            .paragraph("Here's what you can do:")
            .list(["Create workflows", "Run analyses", "View results"])
            .divider()
            .callout("Pro tip: Use natural language!", CalloutType.TIP)
            .code("workflow create rna-seq", language="bash")
            .table(
                headers=["Command", "Description"],
                rows=[
                    ["create", "Create new workflow"],
                    ["run", "Execute workflow"],
                ]
            )
            .progress(75, "Progress")
            .build())
        
        assert len(message) == 8
        
        # Render to different formats
        formatter = MessageFormatter()
        
        markdown = formatter.render(message, MessageFormat.MARKDOWN)
        assert "# Welcome" in markdown
        assert "```bash" in markdown
        
        html = formatter.render(message, MessageFormat.HTML)
        assert "<h1>" in html
        assert "<table>" in html
        
        plain = formatter.render(message, MessageFormat.PLAIN_TEXT)
        assert "WELCOME" in plain
    
    def test_card_with_actions(self):
        """Test building card with actions."""
        card = CardComponent(
            title="Workflow Created",
            subtitle="rna-seq-analysis",
            body="Your RNA-Seq workflow has been created successfully.",
            actions=[
                ButtonComponent(text="Run Now", action="run", style=ButtonStyle.PRIMARY),
                ButtonComponent(text="Edit", action="edit", style=ButtonStyle.SECONDARY),
                ButtonComponent(text="Delete", action="delete", style=ButtonStyle.DANGER),
            ]
        )
        
        formatter = MessageFormatter()
        html = formatter.render([card], MessageFormat.HTML)
        
        assert "Workflow Created" in html
        assert "btn-primary" in html
        assert "btn-danger" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
