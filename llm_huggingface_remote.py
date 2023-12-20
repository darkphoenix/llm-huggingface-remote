import llm
import os
import sys
import click
import json
from typing import Optional, List, Tuple
from text_generation import Client, InferenceAPIClient

try:
    from pydantic import Field, field_validator  # type: ignore
except ImportError:
    from pydantic.class_validators import (
        validator as field_validator,
    )  # type: ignore [no-redef]
    from pydantic.fields import Field

def _ensure_models_file():
    plugin_dir = llm.user_dir() / "huggingface_remote"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    filepath = plugin_dir / "models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath

def build_prompt_blocks_and_system(
    self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]
) -> Tuple[List[str], str]:
    blocks = []

    # Simplified handling of system prompts: use the one from prompt.system, or the
    # one from the first message in the conversation, or the default for the model.
    # Ignore the case where the system prompt changed mid-conversation.
    system_prompt = None
    if prompt.system:
        system_prompt = prompt.system

    if conversation is not None:
        for response in conversation.responses:
            if response.prompt.system:
                system_prompt = response.prompt.system
                break

    if system_prompt is None:
        system_prompt = self.system_prompt()

    template = self.prompt_template()
    # Special case to add <|im_end|> if it looks necessary
    template_end = ""
    if "<|im_start|>" in template and template.count(
        "<|im_start|>"
    ) - 1 == template.count("<|im_end|>"):
        template_end = "<|im_end|>"

    if conversation is not None:
        for prev_response in conversation.responses:
            blocks.append(template.replace("%1", prev_response.prompt.prompt))
            blocks.append(prev_response.text() + template_end)

    # Add the user's prompt
    blocks.append(template.replace("%1", prompt.prompt))

    return blocks, system_prompt

@llm.hookimpl
def register_models(register):
    plugin_dir = llm.user_dir() / "huggingface_remote"
    models_file = plugin_dir / "models.json"
    if not models_file.exists():
        return
    models = json.loads(models_file.read_text())
    for model_id, details in models.items():
        register(
            HuggingfaceRemoteModel(
                model_id,
                details["url"]
            ),
            aliases=details["aliases"],
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def llm_huggingface_remote():
        "Commands for registering Huggingface remote models with LLM"

    @llm_huggingface_remote.command()
    def models_file():
        "Display the path to the models.json file"
        plugin_dir = llm.user_dir() / "huggingface_remote"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        models_file = plugin_dir / "models.json"
        click.echo(models_file)

    @llm_huggingface_remote.command()
    @click.argument(
        "name"
    )
    @click.argument(
        "url"
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def add_model(name, url, aliases):
        "Register a Huggingface remote model"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        info = {
            "name": name,
            "url": url,
            "aliases": aliases
        }
        models[name] = info
        models_file.write_text(json.dumps(models, indent=2))

    @llm_huggingface_remote.command()
    def models():
        "List registered Huggingface models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        click.echo(json.dumps(models, indent=2))


class HuggingfaceRemoteModel(llm.Model):
    class Options(llm.Options):
        max_tokens: int = Field(
            description="Max tokens to return, defaults to 20", default=20
        )

    def __init__(self, model_id, url):
            self.model_id = model_id
            self.url = url

    def execute(self, prompt, stream, response, conversation):
        if self.url.startswith("http"):
            client = Client(self.url)
        else:
            client = InferenceAPIClient(self.url)
        yield client.generate(prompt.prompt, max_new_tokens=prompt.options.max_tokens).generated_text
