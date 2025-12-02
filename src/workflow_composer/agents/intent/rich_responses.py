"""
Rich Response Helpers for Chat Agent.

Phase 6 of Professional Chat Agent implementation.

Features:
- Message formatting utilities
- Interactive component builders
- Multi-modal response support
- Accessibility helpers
- Platform-specific adapters
"""

import json
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


class MessageFormat(Enum):
    """Output format for messages."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"


class ComponentType(Enum):
    """Types of rich components."""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    LINK = "link"
    BUTTON = "button"
    CARD = "card"
    DIVIDER = "divider"
    QUOTE = "quote"
    CALLOUT = "callout"
    PROGRESS = "progress"
    CHART = "chart"
    FORM = "form"
    INPUT = "input"
    CAROUSEL = "carousel"


class CalloutType(Enum):
    """Types of callout boxes."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TIP = "tip"
    NOTE = "note"


class ButtonStyle(Enum):
    """Button visual styles."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    DANGER = "danger"
    WARNING = "warning"
    LINK = "link"


class ListStyle(Enum):
    """List styling options."""
    BULLET = "bullet"
    NUMBERED = "numbered"
    CHECKLIST = "checklist"
    NONE = "none"


@dataclass
class RichComponent:
    """Base class for rich message components."""
    type: ComponentType
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class TextComponent(RichComponent):
    """Plain text component."""
    type: ComponentType = ComponentType.TEXT
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    code: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "bold": self.bold,
            "italic": self.italic,
            "strikethrough": self.strikethrough,
            "code": self.code,
        })
        return d


@dataclass
class HeadingComponent(RichComponent):
    """Heading component."""
    type: ComponentType = ComponentType.HEADING
    level: int = 1  # 1-6
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["level"] = self.level
        return d


@dataclass
class ListComponent(RichComponent):
    """List component."""
    type: ComponentType = ComponentType.LIST
    items: List[str] = field(default_factory=list)
    style: ListStyle = ListStyle.BULLET
    checked: Dict[int, bool] = field(default_factory=dict)  # For checklists
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "items": self.items,
            "style": self.style.value,
            "checked": self.checked,
        })
        return d


@dataclass
class CodeComponent(RichComponent):
    """Code block component."""
    type: ComponentType = ComponentType.CODE
    language: str = ""
    filename: str = ""
    line_numbers: bool = False
    highlight_lines: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "language": self.language,
            "filename": self.filename,
            "line_numbers": self.line_numbers,
            "highlight_lines": self.highlight_lines,
        })
        return d


@dataclass
class TableComponent(RichComponent):
    """Table component."""
    type: ComponentType = ComponentType.TABLE
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""
    sortable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption,
            "sortable": self.sortable,
        })
        return d


@dataclass
class ImageComponent(RichComponent):
    """Image component."""
    type: ComponentType = ComponentType.IMAGE
    url: str = ""
    alt_text: str = ""
    caption: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "url": self.url,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "width": self.width,
            "height": self.height,
        })
        return d


@dataclass
class LinkComponent(RichComponent):
    """Hyperlink component."""
    type: ComponentType = ComponentType.LINK
    url: str = ""
    text: str = ""
    title: str = ""
    external: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "url": self.url,
            "text": self.text or self.url,
            "title": self.title,
            "external": self.external,
        })
        return d


@dataclass
class ButtonComponent(RichComponent):
    """Interactive button component."""
    type: ComponentType = ComponentType.BUTTON
    text: str = ""
    action: str = ""  # Action identifier
    style: ButtonStyle = ButtonStyle.PRIMARY
    disabled: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "action": self.action,
            "style": self.style.value,
            "disabled": self.disabled,
            "data": self.data,
        })
        return d


@dataclass
class CardComponent(RichComponent):
    """Card layout component."""
    type: ComponentType = ComponentType.CARD
    title: str = ""
    subtitle: str = ""
    body: str = ""
    image_url: str = ""
    footer: str = ""
    actions: List[ButtonComponent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "image_url": self.image_url,
            "footer": self.footer,
            "actions": [a.to_dict() for a in self.actions],
        })
        return d


@dataclass
class CalloutComponent(RichComponent):
    """Callout/alert box component."""
    type: ComponentType = ComponentType.CALLOUT
    callout_type: CalloutType = CalloutType.INFO
    title: str = ""
    dismissible: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "callout_type": self.callout_type.value,
            "title": self.title,
            "dismissible": self.dismissible,
        })
        return d


@dataclass
class ProgressComponent(RichComponent):
    """Progress indicator component."""
    type: ComponentType = ComponentType.PROGRESS
    value: float = 0.0  # 0-100
    max_value: float = 100.0
    label: str = ""
    show_percentage: bool = True
    animated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "value": self.value,
            "max_value": self.max_value,
            "label": self.label,
            "show_percentage": self.show_percentage,
            "animated": self.animated,
        })
        return d


@dataclass
class QuoteComponent(RichComponent):
    """Blockquote component."""
    type: ComponentType = ComponentType.QUOTE
    author: str = ""
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "author": self.author,
            "source": self.source,
        })
        return d


@dataclass 
class CarouselComponent(RichComponent):
    """Image/card carousel component."""
    type: ComponentType = ComponentType.CAROUSEL
    items: List[CardComponent] = field(default_factory=list)
    auto_play: bool = False
    interval: int = 5000  # milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "items": [i.to_dict() for i in self.items],
            "auto_play": self.auto_play,
            "interval": self.interval,
        })
        return d


@dataclass
class FormComponent(RichComponent):
    """Form container component."""
    type: ComponentType = ComponentType.FORM
    action: str = ""
    fields: List["InputComponent"] = field(default_factory=list)
    submit_text: str = "Submit"
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "action": self.action,
            "fields": [f.to_dict() for f in self.fields],
            "submit_text": self.submit_text,
        })
        return d


@dataclass
class InputComponent(RichComponent):
    """Form input component."""
    type: ComponentType = ComponentType.INPUT
    input_type: str = "text"  # text, email, number, select, etc.
    name: str = ""
    label: str = ""
    placeholder: str = ""
    required: bool = False
    options: List[str] = field(default_factory=list)  # For select
    default_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "input_type": self.input_type,
            "name": self.name,
            "label": self.label,
            "placeholder": self.placeholder,
            "required": self.required,
            "options": self.options,
            "default_value": self.default_value,
        })
        return d


class MessageBuilder:
    """
    Fluent builder for constructing rich messages.
    
    Example:
        message = (MessageBuilder()
            .heading("Welcome!")
            .text("Here's what you can do:")
            .list(["Create workflows", "Run analyses", "View results"])
            .callout("Pro tip: Use natural language!", CalloutType.TIP)
            .build())
    """
    
    def __init__(self):
        self._components: List[RichComponent] = []
    
    def add(self, component: RichComponent) -> "MessageBuilder":
        """Add a component to the message."""
        self._components.append(component)
        return self
    
    def text(
        self,
        content: str,
        bold: bool = False,
        italic: bool = False,
        code: bool = False
    ) -> "MessageBuilder":
        """Add text content."""
        self._components.append(TextComponent(
            content=content,
            bold=bold,
            italic=italic,
            code=code
        ))
        return self
    
    def heading(self, content: str, level: int = 1) -> "MessageBuilder":
        """Add a heading."""
        self._components.append(HeadingComponent(
            content=content,
            level=level
        ))
        return self
    
    def paragraph(self, content: str) -> "MessageBuilder":
        """Add a paragraph."""
        self._components.append(RichComponent(
            type=ComponentType.PARAGRAPH,
            content=content
        ))
        return self
    
    def list(
        self,
        items: List[str],
        style: ListStyle = ListStyle.BULLET,
        checked: Optional[Dict[int, bool]] = None
    ) -> "MessageBuilder":
        """Add a list."""
        self._components.append(ListComponent(
            items=items,
            style=style,
            checked=checked or {}
        ))
        return self
    
    def code(
        self,
        content: str,
        language: str = "",
        filename: str = "",
        line_numbers: bool = False
    ) -> "MessageBuilder":
        """Add a code block."""
        self._components.append(CodeComponent(
            content=content,
            language=language,
            filename=filename,
            line_numbers=line_numbers
        ))
        return self
    
    def table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: str = ""
    ) -> "MessageBuilder":
        """Add a table."""
        self._components.append(TableComponent(
            headers=headers,
            rows=rows,
            caption=caption
        ))
        return self
    
    def image(
        self,
        url: str,
        alt_text: str = "",
        caption: str = ""
    ) -> "MessageBuilder":
        """Add an image."""
        self._components.append(ImageComponent(
            url=url,
            alt_text=alt_text,
            caption=caption
        ))
        return self
    
    def link(
        self,
        url: str,
        text: Optional[str] = None,
        external: bool = False
    ) -> "MessageBuilder":
        """Add a hyperlink."""
        self._components.append(LinkComponent(
            url=url,
            text=text or url,
            external=external
        ))
        return self
    
    def button(
        self,
        text: str,
        action: str,
        style: ButtonStyle = ButtonStyle.PRIMARY,
        data: Optional[Dict[str, Any]] = None
    ) -> "MessageBuilder":
        """Add a button."""
        self._components.append(ButtonComponent(
            text=text,
            action=action,
            style=style,
            data=data or {}
        ))
        return self
    
    def card(
        self,
        title: str,
        body: str = "",
        subtitle: str = "",
        image_url: str = "",
        actions: Optional[List[ButtonComponent]] = None
    ) -> "MessageBuilder":
        """Add a card."""
        self._components.append(CardComponent(
            title=title,
            body=body,
            subtitle=subtitle,
            image_url=image_url,
            actions=actions or []
        ))
        return self
    
    def callout(
        self,
        content: str,
        callout_type: CalloutType = CalloutType.INFO,
        title: str = ""
    ) -> "MessageBuilder":
        """Add a callout box."""
        self._components.append(CalloutComponent(
            content=content,
            callout_type=callout_type,
            title=title
        ))
        return self
    
    def progress(
        self,
        value: float,
        label: str = "",
        max_value: float = 100.0
    ) -> "MessageBuilder":
        """Add a progress indicator."""
        self._components.append(ProgressComponent(
            value=value,
            label=label,
            max_value=max_value
        ))
        return self
    
    def quote(
        self,
        content: str,
        author: str = "",
        source: str = ""
    ) -> "MessageBuilder":
        """Add a blockquote."""
        self._components.append(QuoteComponent(
            content=content,
            author=author,
            source=source
        ))
        return self
    
    def divider(self) -> "MessageBuilder":
        """Add a divider/separator."""
        self._components.append(RichComponent(type=ComponentType.DIVIDER))
        return self
    
    def newline(self) -> "MessageBuilder":
        """Add a line break."""
        self._components.append(TextComponent(content="\n"))
        return self
    
    def build(self) -> List[RichComponent]:
        """Build and return the list of components."""
        return self._components.copy()
    
    def clear(self) -> "MessageBuilder":
        """Clear all components."""
        self._components.clear()
        return self


class FormatAdapter(ABC):
    """Abstract base class for format adapters."""
    
    @property
    @abstractmethod
    def format(self) -> MessageFormat:
        """Get the format this adapter produces."""
        pass
    
    @abstractmethod
    def render(self, components: List[RichComponent]) -> str:
        """Render components to formatted string."""
        pass
    
    @abstractmethod
    def render_component(self, component: RichComponent) -> str:
        """Render a single component."""
        pass


class PlainTextAdapter(FormatAdapter):
    """Adapter for plain text output."""
    
    @property
    def format(self) -> MessageFormat:
        return MessageFormat.PLAIN_TEXT
    
    def render(self, components: List[RichComponent]) -> str:
        """Render components to plain text."""
        parts = []
        for component in components:
            rendered = self.render_component(component)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)
    
    def render_component(self, component: RichComponent) -> str:
        """Render a single component to plain text."""
        if isinstance(component, TextComponent):
            return component.content or ""
        
        elif isinstance(component, HeadingComponent):
            return f"\n{component.content.upper()}\n{'=' * len(component.content)}"
        
        elif isinstance(component, ListComponent):
            lines = []
            for i, item in enumerate(component.items, 1):
                if component.style == ListStyle.NUMBERED:
                    lines.append(f"{i}. {item}")
                elif component.style == ListStyle.CHECKLIST:
                    checked = component.checked.get(i-1, False)
                    mark = "[x]" if checked else "[ ]"
                    lines.append(f"{mark} {item}")
                else:
                    lines.append(f"• {item}")
            return "\n".join(lines)
        
        elif isinstance(component, CodeComponent):
            lines = ["---CODE---"]
            if component.filename:
                lines.append(f"File: {component.filename}")
            lines.append(component.content or "")
            lines.append("-----------")
            return "\n".join(lines)
        
        elif isinstance(component, TableComponent):
            if not component.headers and not component.rows:
                return ""
            
            lines = []
            if component.caption:
                lines.append(f"Table: {component.caption}")
            
            # Headers
            if component.headers:
                lines.append(" | ".join(component.headers))
                lines.append("-" * (sum(len(h) for h in component.headers) + 3 * (len(component.headers) - 1)))
            
            # Rows
            for row in component.rows:
                lines.append(" | ".join(str(cell) for cell in row))
            
            return "\n".join(lines)
        
        elif isinstance(component, ImageComponent):
            return f"[Image: {component.alt_text or component.url}]"
        
        elif isinstance(component, LinkComponent):
            return f"{component.text or component.url} ({component.url})"
        
        elif isinstance(component, ButtonComponent):
            return f"[{component.text}]"
        
        elif isinstance(component, CardComponent):
            lines = [f"┌─ {component.title} ─┐"]
            if component.subtitle:
                lines.append(f"  {component.subtitle}")
            if component.body:
                lines.append(f"  {component.body}")
            if component.actions:
                lines.append("  Actions: " + " | ".join(a.text for a in component.actions))
            lines.append("└" + "─" * (len(component.title) + 4) + "┘")
            return "\n".join(lines)
        
        elif isinstance(component, CalloutComponent):
            icon = {
                CalloutType.INFO: "ℹ",
                CalloutType.SUCCESS: "✓",
                CalloutType.WARNING: "⚠",
                CalloutType.ERROR: "✗",
                CalloutType.TIP: "💡",
                CalloutType.NOTE: "📝",
            }.get(component.callout_type, "•")
            title = f"{component.title}: " if component.title else ""
            return f"{icon} {title}{component.content}"
        
        elif isinstance(component, ProgressComponent):
            percentage = (component.value / component.max_value) * 100
            filled = int(percentage / 5)
            bar = "█" * filled + "░" * (20 - filled)
            label = f"{component.label}: " if component.label else ""
            pct = f" {percentage:.0f}%" if component.show_percentage else ""
            return f"{label}[{bar}]{pct}"
        
        elif isinstance(component, QuoteComponent):
            lines = [f'"{component.content}"']
            if component.author:
                lines.append(f"  — {component.author}")
            return "\n".join(lines)
        
        elif component.type == ComponentType.DIVIDER:
            return "\n" + "─" * 40 + "\n"
        
        elif component.type == ComponentType.PARAGRAPH:
            return component.content or ""
        
        return str(component.content or "")


class MarkdownAdapter(FormatAdapter):
    """Adapter for Markdown output."""
    
    @property
    def format(self) -> MessageFormat:
        return MessageFormat.MARKDOWN
    
    def render(self, components: List[RichComponent]) -> str:
        """Render components to Markdown."""
        parts = []
        for component in components:
            rendered = self.render_component(component)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)
    
    def render_component(self, component: RichComponent) -> str:
        """Render a single component to Markdown."""
        if isinstance(component, TextComponent):
            text = component.content or ""
            if component.bold:
                text = f"**{text}**"
            if component.italic:
                text = f"*{text}*"
            if component.strikethrough:
                text = f"~~{text}~~"
            if component.code:
                text = f"`{text}`"
            return text
        
        elif isinstance(component, HeadingComponent):
            return "#" * component.level + " " + (component.content or "")
        
        elif isinstance(component, ListComponent):
            lines = []
            for i, item in enumerate(component.items, 1):
                if component.style == ListStyle.NUMBERED:
                    lines.append(f"{i}. {item}")
                elif component.style == ListStyle.CHECKLIST:
                    checked = component.checked.get(i-1, False)
                    mark = "[x]" if checked else "[ ]"
                    lines.append(f"- {mark} {item}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        
        elif isinstance(component, CodeComponent):
            lang = component.language or ""
            code = component.content or ""
            return f"```{lang}\n{code}\n```"
        
        elif isinstance(component, TableComponent):
            if not component.headers:
                return ""
            
            lines = []
            # Headers
            lines.append("| " + " | ".join(component.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(component.headers)) + " |")
            
            # Rows
            for row in component.rows:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            if component.caption:
                lines.append(f"\n*{component.caption}*")
            
            return "\n".join(lines)
        
        elif isinstance(component, ImageComponent):
            alt = component.alt_text or "image"
            caption = f"\n*{component.caption}*" if component.caption else ""
            return f"![{alt}]({component.url}){caption}"
        
        elif isinstance(component, LinkComponent):
            text = component.text or component.url
            return f"[{text}]({component.url})"
        
        elif isinstance(component, ButtonComponent):
            # Markdown doesn't support buttons, render as link
            return f"**[{component.text}]**"
        
        elif isinstance(component, CardComponent):
            lines = []
            if component.title:
                lines.append(f"### {component.title}")
            if component.subtitle:
                lines.append(f"*{component.subtitle}*")
            if component.image_url:
                lines.append(f"![{component.title}]({component.image_url})")
            if component.body:
                lines.append(component.body)
            if component.actions:
                lines.append(" | ".join(f"**[{a.text}]**" for a in component.actions))
            return "\n\n".join(lines)
        
        elif isinstance(component, CalloutComponent):
            # Use blockquote with emoji
            icon = {
                CalloutType.INFO: "ℹ️",
                CalloutType.SUCCESS: "✅",
                CalloutType.WARNING: "⚠️",
                CalloutType.ERROR: "❌",
                CalloutType.TIP: "💡",
                CalloutType.NOTE: "📝",
            }.get(component.callout_type, "•")
            title = f"**{component.title}**\n" if component.title else ""
            return f"> {icon} {title}> {component.content}"
        
        elif isinstance(component, ProgressComponent):
            percentage = (component.value / component.max_value) * 100
            label = f"**{component.label}**: " if component.label else ""
            return f"{label}{percentage:.0f}%"
        
        elif isinstance(component, QuoteComponent):
            lines = [f"> {component.content}"]
            if component.author:
                lines.append(f">\n> — *{component.author}*")
            return "\n".join(lines)
        
        elif component.type == ComponentType.DIVIDER:
            return "---"
        
        elif component.type == ComponentType.PARAGRAPH:
            return component.content or ""
        
        return str(component.content or "")


class HTMLAdapter(FormatAdapter):
    """Adapter for HTML output."""
    
    @property
    def format(self) -> MessageFormat:
        return MessageFormat.HTML
    
    def render(self, components: List[RichComponent]) -> str:
        """Render components to HTML."""
        parts = []
        for component in components:
            rendered = self.render_component(component)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)
    
    def _escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))
    
    def render_component(self, component: RichComponent) -> str:
        """Render a single component to HTML."""
        if isinstance(component, TextComponent):
            text = self._escape(component.content or "")
            if component.code:
                text = f"<code>{text}</code>"
            if component.bold:
                text = f"<strong>{text}</strong>"
            if component.italic:
                text = f"<em>{text}</em>"
            if component.strikethrough:
                text = f"<del>{text}</del>"
            return f"<span>{text}</span>"
        
        elif isinstance(component, HeadingComponent):
            level = min(max(component.level, 1), 6)
            content = self._escape(component.content or "")
            return f"<h{level}>{content}</h{level}>"
        
        elif isinstance(component, ListComponent):
            if component.style == ListStyle.NUMBERED:
                tag = "ol"
            else:
                tag = "ul"
            
            items = []
            for i, item in enumerate(component.items):
                if component.style == ListStyle.CHECKLIST:
                    checked = "checked" if component.checked.get(i, False) else ""
                    items.append(f'<li><input type="checkbox" {checked} disabled> {self._escape(item)}</li>')
                else:
                    items.append(f"<li>{self._escape(item)}</li>")
            
            return f"<{tag}>\n" + "\n".join(items) + f"\n</{tag}>"
        
        elif isinstance(component, CodeComponent):
            lang_class = f' class="language-{component.language}"' if component.language else ""
            content = self._escape(component.content or "")
            return f"<pre><code{lang_class}>{content}</code></pre>"
        
        elif isinstance(component, TableComponent):
            lines = ['<table>']
            
            if component.caption:
                lines.append(f"<caption>{self._escape(component.caption)}</caption>")
            
            if component.headers:
                lines.append("<thead><tr>")
                for header in component.headers:
                    lines.append(f"<th>{self._escape(header)}</th>")
                lines.append("</tr></thead>")
            
            lines.append("<tbody>")
            for row in component.rows:
                lines.append("<tr>")
                for cell in row:
                    lines.append(f"<td>{self._escape(str(cell))}</td>")
                lines.append("</tr>")
            lines.append("</tbody>")
            lines.append("</table>")
            
            return "\n".join(lines)
        
        elif isinstance(component, ImageComponent):
            alt = self._escape(component.alt_text or "")
            attrs = [f'src="{component.url}"', f'alt="{alt}"']
            if component.width:
                attrs.append(f'width="{component.width}"')
            if component.height:
                attrs.append(f'height="{component.height}"')
            
            img = f'<img {" ".join(attrs)}>'
            if component.caption:
                return f'<figure>{img}<figcaption>{self._escape(component.caption)}</figcaption></figure>'
            return img
        
        elif isinstance(component, LinkComponent):
            text = self._escape(component.text or component.url)
            target = ' target="_blank"' if component.external else ""
            title = f' title="{self._escape(component.title)}"' if component.title else ""
            return f'<a href="{component.url}"{target}{title}>{text}</a>'
        
        elif isinstance(component, ButtonComponent):
            style_class = f'btn-{component.style.value}'
            disabled = " disabled" if component.disabled else ""
            data_attrs = " ".join(f'data-{k}="{v}"' for k, v in component.data.items())
            return f'<button class="btn {style_class}" data-action="{component.action}" {data_attrs}{disabled}>{self._escape(component.text)}</button>'
        
        elif isinstance(component, CardComponent):
            lines = ['<div class="card">']
            if component.image_url:
                lines.append(f'<img class="card-img-top" src="{component.image_url}" alt="{self._escape(component.title)}">')
            lines.append('<div class="card-body">')
            if component.title:
                lines.append(f'<h5 class="card-title">{self._escape(component.title)}</h5>')
            if component.subtitle:
                lines.append(f'<h6 class="card-subtitle">{self._escape(component.subtitle)}</h6>')
            if component.body:
                lines.append(f'<p class="card-text">{self._escape(component.body)}</p>')
            if component.actions:
                for action in component.actions:
                    lines.append(self.render_component(action))
            lines.append("</div>")
            if component.footer:
                lines.append(f'<div class="card-footer">{self._escape(component.footer)}</div>')
            lines.append("</div>")
            return "\n".join(lines)
        
        elif isinstance(component, CalloutComponent):
            alert_class = {
                CalloutType.INFO: "alert-info",
                CalloutType.SUCCESS: "alert-success",
                CalloutType.WARNING: "alert-warning",
                CalloutType.ERROR: "alert-danger",
                CalloutType.TIP: "alert-info",
                CalloutType.NOTE: "alert-secondary",
            }.get(component.callout_type, "alert-info")
            
            title = f'<strong>{self._escape(component.title)}</strong> ' if component.title else ""
            return f'<div class="alert {alert_class}">{title}{self._escape(component.content or "")}</div>'
        
        elif isinstance(component, ProgressComponent):
            percentage = (component.value / component.max_value) * 100
            label = f'<label>{self._escape(component.label)}</label>' if component.label else ""
            pct_text = f'{percentage:.0f}%' if component.show_percentage else ""
            return f'''
{label}
<div class="progress">
  <div class="progress-bar" style="width: {percentage}%">{pct_text}</div>
</div>'''
        
        elif isinstance(component, QuoteComponent):
            author = f'<cite>— {self._escape(component.author)}</cite>' if component.author else ""
            return f'<blockquote>{self._escape(component.content or "")}{author}</blockquote>'
        
        elif component.type == ComponentType.DIVIDER:
            return "<hr>"
        
        elif component.type == ComponentType.PARAGRAPH:
            return f"<p>{self._escape(component.content or '')}</p>"
        
        return f"<span>{self._escape(str(component.content or ''))}</span>"


class SlackAdapter(FormatAdapter):
    """Adapter for Slack message format."""
    
    @property
    def format(self) -> MessageFormat:
        return MessageFormat.SLACK
    
    def render(self, components: List[RichComponent]) -> str:
        """Render components to Slack mrkdwn format."""
        parts = []
        for component in components:
            rendered = self.render_component(component)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)
    
    def render_component(self, component: RichComponent) -> str:
        """Render a single component to Slack format."""
        if isinstance(component, TextComponent):
            text = component.content or ""
            if component.bold:
                text = f"*{text}*"
            if component.italic:
                text = f"_{text}_"
            if component.strikethrough:
                text = f"~{text}~"
            if component.code:
                text = f"`{text}`"
            return text
        
        elif isinstance(component, HeadingComponent):
            # Slack doesn't have headings, use bold
            return f"*{component.content or ''}*"
        
        elif isinstance(component, ListComponent):
            lines = []
            for i, item in enumerate(component.items, 1):
                if component.style == ListStyle.NUMBERED:
                    lines.append(f"{i}. {item}")
                else:
                    lines.append(f"• {item}")
            return "\n".join(lines)
        
        elif isinstance(component, CodeComponent):
            return f"```{component.content or ''}```"
        
        elif isinstance(component, LinkComponent):
            text = component.text or component.url
            return f"<{component.url}|{text}>"
        
        elif isinstance(component, QuoteComponent):
            return f">{component.content or ''}"
        
        elif component.type == ComponentType.DIVIDER:
            return "───────────────"
        
        return str(component.content or "")
    
    def to_blocks(self, components: List[RichComponent]) -> List[Dict[str, Any]]:
        """Convert components to Slack Block Kit format."""
        blocks = []
        
        for component in components:
            if isinstance(component, TextComponent) or component.type == ComponentType.PARAGRAPH:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": self.render_component(component)
                    }
                })
            
            elif isinstance(component, HeadingComponent):
                blocks.append({
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": component.content or ""
                    }
                })
            
            elif isinstance(component, ImageComponent):
                blocks.append({
                    "type": "image",
                    "image_url": component.url,
                    "alt_text": component.alt_text or "image"
                })
            
            elif isinstance(component, ButtonComponent):
                blocks.append({
                    "type": "actions",
                    "elements": [{
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": component.text
                        },
                        "action_id": component.action
                    }]
                })
            
            elif component.type == ComponentType.DIVIDER:
                blocks.append({"type": "divider"})
        
        return blocks


class MessageFormatter:
    """
    Main message formatter that orchestrates rendering.
    
    Supports multiple output formats and provides
    convenient methods for common formatting tasks.
    """
    
    def __init__(self, default_format: MessageFormat = MessageFormat.MARKDOWN):
        self.default_format = default_format
        self._adapters: Dict[MessageFormat, FormatAdapter] = {
            MessageFormat.PLAIN_TEXT: PlainTextAdapter(),
            MessageFormat.MARKDOWN: MarkdownAdapter(),
            MessageFormat.HTML: HTMLAdapter(),
            MessageFormat.SLACK: SlackAdapter(),
        }
    
    def register_adapter(self, adapter: FormatAdapter) -> None:
        """Register a custom adapter."""
        self._adapters[adapter.format] = adapter
    
    def render(
        self,
        components: List[RichComponent],
        format: Optional[MessageFormat] = None
    ) -> str:
        """Render components to formatted string."""
        format = format or self.default_format
        adapter = self._adapters.get(format)
        
        if not adapter:
            raise ValueError(f"No adapter registered for format: {format}")
        
        return adapter.render(components)
    
    def builder(self) -> MessageBuilder:
        """Get a new message builder."""
        return MessageBuilder()
    
    # Convenience methods
    def format_text(
        self,
        text: str,
        bold: bool = False,
        italic: bool = False,
        code: bool = False,
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a text string."""
        component = TextComponent(content=text, bold=bold, italic=italic, code=code)
        return self.render([component], format)
    
    def format_list(
        self,
        items: List[str],
        style: ListStyle = ListStyle.BULLET,
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a list."""
        component = ListComponent(items=items, style=style)
        return self.render([component], format)
    
    def format_code(
        self,
        code: str,
        language: str = "",
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format code."""
        component = CodeComponent(content=code, language=language)
        return self.render([component], format)
    
    def format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a table."""
        component = TableComponent(headers=headers, rows=rows)
        return self.render([component], format)
    
    def format_error(
        self,
        message: str,
        details: str = "",
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format an error message."""
        components = [
            CalloutComponent(
                content=message,
                callout_type=CalloutType.ERROR,
                title="Error"
            )
        ]
        if details:
            components.append(CodeComponent(content=details))
        return self.render(components, format)
    
    def format_success(
        self,
        message: str,
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a success message."""
        component = CalloutComponent(
            content=message,
            callout_type=CalloutType.SUCCESS,
            title="Success"
        )
        return self.render([component], format)
    
    def format_warning(
        self,
        message: str,
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a warning message."""
        component = CalloutComponent(
            content=message,
            callout_type=CalloutType.WARNING,
            title="Warning"
        )
        return self.render([component], format)
    
    def format_progress(
        self,
        current: int,
        total: int,
        label: str = "",
        format: Optional[MessageFormat] = None
    ) -> str:
        """Format a progress indicator."""
        percentage = (current / total * 100) if total > 0 else 0
        component = ProgressComponent(
            value=percentage,
            label=label or f"{current}/{total}"
        )
        return self.render([component], format)


# Utility functions
def strip_formatting(text: str) -> str:
    """Remove all formatting from text."""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.+?)__', r'\1', text)      # Bold
    text = re.sub(r'_(.+?)_', r'\1', text)        # Italic
    text = re.sub(r'~~(.+?)~~', r'\1', text)      # Strikethrough
    text = re.sub(r'`(.+?)`', r'\1', text)        # Code
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links
    return text


def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(ellipsis)] + ellipsis


def word_wrap(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


def format_number(value: float, precision: int = 2) -> str:
    """Format number with thousand separators."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.{precision}f}M"
    elif value >= 1_000:
        return f"{value/1_000:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_value) < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} PB"


# Singleton formatter
_formatter: Optional[MessageFormatter] = None


def get_message_formatter() -> MessageFormatter:
    """Get the singleton message formatter."""
    global _formatter
    if _formatter is None:
        _formatter = MessageFormatter()
    return _formatter


def reset_message_formatter() -> None:
    """Reset the singleton formatter."""
    global _formatter
    _formatter = None


__all__ = [
    # Enums
    "MessageFormat",
    "ComponentType",
    "CalloutType",
    "ButtonStyle",
    "ListStyle",
    
    # Components
    "RichComponent",
    "TextComponent",
    "HeadingComponent",
    "ListComponent",
    "CodeComponent",
    "TableComponent",
    "ImageComponent",
    "LinkComponent",
    "ButtonComponent",
    "CardComponent",
    "CalloutComponent",
    "ProgressComponent",
    "QuoteComponent",
    "CarouselComponent",
    "FormComponent",
    "InputComponent",
    
    # Builder
    "MessageBuilder",
    
    # Adapters
    "FormatAdapter",
    "PlainTextAdapter",
    "MarkdownAdapter",
    "HTMLAdapter",
    "SlackAdapter",
    
    # Formatter
    "MessageFormatter",
    
    # Utilities
    "strip_formatting",
    "truncate_text",
    "word_wrap",
    "format_duration",
    "format_number",
    "format_bytes",
    
    # Singleton
    "get_message_formatter",
    "reset_message_formatter",
]
