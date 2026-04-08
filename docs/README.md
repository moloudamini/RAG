# Documentation

## Architecture Diagram

The `architecture-diagram.svg` file contains the visual representation of the RAG system architecture.

### Converting to PNG (Optional)

If you prefer a PNG version for better compatibility:

**Using Inkscape:**
```bash
inkscape docs/architecture-diagram.svg --export-filename=docs/architecture-diagram.png --export-dpi=150
```

**Using ImageMagick:**
```bash
magick docs/architecture-diagram.svg docs/architecture-diagram.png
```

**Online Converters:**
- Upload the SVG to any online SVG-to-PNG converter
- GitHub will automatically display SVG files in README.md

### Diagram Components

- **Blue boxes**: LangGraph agents (Q&A and Analytics)
- **Purple boxes**: Processing steps
- **Green boxes**: Evaluation and response generation
- **Arrows**: Data flow through the system

The diagram shows how user queries are automatically classified and routed to the appropriate specialized agent for processing.