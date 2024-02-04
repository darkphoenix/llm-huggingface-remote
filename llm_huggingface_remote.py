import llm
import os
import sys
import click
import json
from typing import Optional, List, Tuple
from text_generation import Client, InferenceAPIClient

DEFAULT_SYSTEM_PROMPT = """###System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
DEFAULT_PROMPT_TEMPLATE = "%SYS### Human: \n%1\n### Assistant:\n%2"
#DEFAULT_PROMPT_TEMPLATE = "<s>[INST] %SYS %1 [/INST] %2 </s>"

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
                details["url"],
                details['use_chat_prompt']
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
    @click.option(
        "--chat",
        is_flag=True,
        help="Do not use a chat prompt format.",
    )
    def add_model(name, url, aliases, chat):
        "Register a Huggingface remote model"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        info = {
            "name": name,
            "url": url,
            "aliases": aliases,
            "use_chat_prompt": chat
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
            description="Max tokens to return, defaults to 200", default=200
        )

    def __init__(self, model_id, url, use_chat_prompt, prompt_template=None, system_prompt=None):
            self.model_id = model_id
            self.url = url
            self.use_chat_prompt = use_chat_prompt
            self.prompt_template = prompt_template
            self.system_prompt = system_prompt
    
    def get_prompt_template(self):
        return (
            self.prompt_template or DEFAULT_PROMPT_TEMPLATE
        )

    def get_system_prompt(self):
        return (
            self.system_prompt or DEFAULT_SYSTEM_PROMPT
        )

    def build_prompt_blocks(
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
            system_prompt = self.get_system_prompt()

        if conversation is not None:
            first = True
            for prev_response in conversation.responses:
                template = self.get_prompt_template()
                if first:
                    template = template.replace("%SYS", system_prompt + "\n")
                    first = False
                else:
                    template = template.replace("%SYS", "")
                template = template.replace("%1", prev_response.prompt.prompt)
                template = template.replace("%2", prev_response.text())
                blocks.append(template)

        # Add the user's prompt
        template = self.get_prompt_template()
        if not blocks:
            template = template.replace("%SYS", system_prompt + "\n")
        else:
            template = template.replace("%SYS", "")
        blocks.append(template.replace("%1", prompt.prompt).split("%2")[0])

        return blocks

    def execute(self, prompt, stream, response, conversation):
        if self.url.startswith("http"):
            client = Client(self.url)
        else:
            client = InferenceAPIClient(self.url)
        
        #Assemble prompt from history
        if self.use_chat_prompt:
            blocks = self.build_prompt_blocks(prompt, conversation)
            text_prompt = "".join(blocks)
        else:
            text_prompt = prompt.prompt
        
        yield client.generate(text_prompt, max_new_tokens=prompt.options.max_tokens).generated_text[len(text_prompt):]
