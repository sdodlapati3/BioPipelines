#!/usr/bin/env python3
"""
BioPipelines Workflow Composer CLI
==================================

Command-line interface for the AI Workflow Composer.

Usage:
    biocomposer generate "RNA-seq differential expression, mouse"
    biocomposer chat --llm ollama
    biocomposer tools search star
    biocomposer modules list
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow_composer import Composer, Config
from workflow_composer.llm import get_llm, list_providers, check_providers


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def cmd_generate(args):
    """Generate a workflow from description."""
    setup_logging(args.verbose)
    
    # Initialize composer
    llm = get_llm(args.llm, args.model) if args.llm else None
    composer = Composer(llm=llm)
    
    # Generate workflow
    workflow = composer.generate(
        args.description,
        output_dir=args.output,
        auto_create_modules=not args.no_auto_create
    )
    
    if args.output:
        print(f"\n✓ Workflow saved to: {args.output}")
    else:
        print(f"\n✓ Workflow generated: {workflow.name}")
        if args.show:
            print("\n" + "="*60)
            print("main.nf:")
            print("="*60)
            print(workflow.main_nf)


def cmd_chat(args):
    """Interactive chat mode."""
    setup_logging(args.verbose)
    
    # Initialize composer
    llm = get_llm(args.llm, args.model) if args.llm else None
    composer = Composer(llm=llm)
    
    print("BioPipelines Workflow Composer - Interactive Mode")
    print(f"Using LLM: {composer.llm}")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "help":
            print("""
Commands:
  generate <description>  - Generate a workflow
  parse <description>     - Parse intent only
  tools <name>           - Search for tools
  modules                - List available modules
  stats                  - Show system stats
  switch <provider>      - Switch LLM provider
  quit                   - Exit
""")
            continue
        
        if user_input.lower().startswith("switch "):
            provider = user_input[7:].strip()
            composer.switch_llm(provider)
            print(f"Switched to: {composer.llm}")
            continue
        
        if user_input.lower() == "stats":
            stats = composer.get_stats()
            print(json.dumps(stats, indent=2, default=str))
            continue
        
        if user_input.lower() == "modules":
            modules = composer.module_mapper.list_by_category()
            for cat, mods in modules.items():
                print(f"\n{cat}:")
                for m in mods:
                    print(f"  - {m}")
            continue
        
        if user_input.lower().startswith("tools "):
            query = user_input[6:].strip()
            matches = composer.tool_selector.fuzzy_search(query)
            print(f"\nTools matching '{query}':")
            for match in matches[:10]:
                print(f"  - {match.tool.name} ({match.tool.container}) - score: {match.score:.2f}")
            continue
        
        if user_input.lower().startswith("parse "):
            desc = user_input[6:].strip()
            intent = composer.parse_intent(desc)
            print(f"\nParsed Intent:")
            print(json.dumps(intent.to_dict(), indent=2))
            continue
        
        # Default: try to generate workflow
        print("\nAssistant: Analyzing your request...")
        
        try:
            # Check readiness first
            readiness = composer.check_readiness(user_input)
            
            if not readiness["ready"]:
                print("Issues detected:")
                for issue in readiness["issues"]:
                    print(f"  - {issue}")
                continue
            
            if readiness["warnings"]:
                print("Warnings:")
                for warning in readiness["warnings"]:
                    print(f"  - {warning}")
            
            print(f"\nTools found: {readiness['tools_found']}")
            print(f"Modules found: {readiness['modules_found']}")
            
            if readiness["modules_missing"]:
                print(f"Missing modules: {', '.join(readiness['modules_missing'])}")
            
            proceed = input("\nGenerate workflow? (y/n): ").strip().lower()
            if proceed == 'y':
                workflow = composer.generate(user_input)
                output_dir = input("Save to directory (or Enter to skip): ").strip()
                if output_dir:
                    workflow.save(output_dir)
                    print(f"✓ Saved to {output_dir}")
                else:
                    print("✓ Workflow generated (not saved)")
        
        except Exception as e:
            print(f"Error: {e}")


def cmd_tools(args):
    """Search or list tools."""
    setup_logging(args.verbose)
    
    composer = Composer()
    
    if args.search:
        matches = composer.tool_selector.fuzzy_search(args.search)
        print(f"Tools matching '{args.search}':")
        for match in matches[:20]:
            print(f"  {match.tool.name:20} ({match.tool.container:15}) score: {match.score:.2f}")
    
    elif args.container:
        tools = composer.tool_selector.get_tools_in_container(args.container)
        print(f"Tools in {args.container}:")
        for tool in sorted(tools, key=lambda t: t.name)[:50]:
            print(f"  {tool.name}")
        if len(tools) > 50:
            print(f"  ... and {len(tools) - 50} more")
    
    else:
        stats = composer.tool_selector.get_stats()
        print(f"Total tools: {stats['total_tools']}")
        print(f"Containers: {stats['containers']}")
        print("\nTools per container:")
        for name, count in stats['tools_per_container'].items():
            print(f"  {name}: {count}")


def cmd_modules(args):
    """List or search modules."""
    setup_logging(args.verbose)
    
    composer = Composer()
    
    if args.list:
        modules = composer.module_mapper.list_by_category()
        for category, mods in sorted(modules.items()):
            print(f"\n{category}:")
            for mod in sorted(mods):
                print(f"  - {mod}")
    
    elif args.find:
        module = composer.module_mapper.find_module(args.find)
        if module:
            print(f"Module: {module.name}")
            print(f"Path: {module.path}")
            print(f"Container: {module.container}")
            print(f"Processes: {', '.join(module.processes)}")
        else:
            print(f"Module not found: {args.find}")
    
    else:
        print(f"Total modules: {len(composer.module_mapper.modules)}")
        modules = composer.module_mapper.list_by_category()
        for category, mods in sorted(modules.items()):
            print(f"  {category}: {len(mods)}")


def cmd_providers(args):
    """List and check LLM providers."""
    setup_logging(args.verbose)
    
    print("Available LLM Providers:")
    providers = list_providers()
    for name, cls in providers.items():
        print(f"  {name}: {cls}")
    
    if args.check:
        print("\nAvailability check:")
        status = check_providers()
        for name, available in status.items():
            status_str = "✓" if available else "✗"
            print(f"  {status_str} {name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="biocomposer",
        description="BioPipelines AI Workflow Composer"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a workflow")
    gen_parser.add_argument("description", help="Natural language description")
    gen_parser.add_argument("-o", "--output", help="Output directory")
    gen_parser.add_argument("-l", "--llm", help="LLM provider (ollama, openai, anthropic)")
    gen_parser.add_argument("-m", "--model", help="Model name")
    gen_parser.add_argument("--no-auto-create", action="store_true", help="Don't auto-create missing modules")
    gen_parser.add_argument("--show", action="store_true", help="Show generated code")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("-l", "--llm", help="LLM provider")
    chat_parser.add_argument("-m", "--model", help="Model name")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Tools command
    tools_parser = subparsers.add_parser("tools", help="Search tools")
    tools_parser.add_argument("-s", "--search", help="Search query")
    tools_parser.add_argument("-c", "--container", help="List tools in container")
    tools_parser.set_defaults(func=cmd_tools)
    
    # Modules command
    modules_parser = subparsers.add_parser("modules", help="List modules")
    modules_parser.add_argument("-l", "--list", action="store_true", help="List all modules")
    modules_parser.add_argument("-f", "--find", help="Find specific module")
    modules_parser.set_defaults(func=cmd_modules)
    
    # Providers command
    prov_parser = subparsers.add_parser("providers", help="List LLM providers")
    prov_parser.add_argument("-c", "--check", action="store_true", help="Check availability")
    prov_parser.set_defaults(func=cmd_providers)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
