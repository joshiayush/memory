"""Prompt templates for the AgenticChunker.

This module contains all LLM prompt templates used by the AgenticChunker class
for semantic chunking operations. Prompts are defined as ChatPromptTemplate
objects with system and user message pairs.
"""

import textwrap

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


# System message for chunk summary updates
_UPDATE_SUMMARY_SYSTEM = """You are the steward of a group of chunks which
represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate
a very brief 1-sentence summary which will inform viewers what a chunk group
is about.

A good summary will say what the chunk is about, and give any clarifying
instructions on what to add to the chunk.

You will be given a group of propositions which are in the chunk and the
chunk's current summary.

Your summaries should anticipate generalization. If you get a proposition
about apples, generalize it to food. Or month, generalize it to
"date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes
to eat.

Only respond with the chunk new summary, nothing else."""

UPDATE_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_UPDATE_SUMMARY_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Chunk's propositions:
                {propositions}

                Current chunk summary:
                {current_summary}
                """
            )
        ),
    ]
)
"""Prompt for updating an existing chunk's summary after adding a proposition.

Template Variables:
    propositions: Newline-separated list of propositions in the chunk.
    current_summary: The chunk's current summary text.
"""


# System message for chunk title updates
_UPDATE_TITLE_SYSTEM = """You are the steward of a group of chunks which
represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate
a very brief updated chunk title which will inform viewers what a chunk
group is about.

A good title will say what the chunk is about.

You will be given a group of propositions which are in the chunk, chunk
summary and the chunk title.

Your title should anticipate generalization. If you get a proposition about
apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else."""

UPDATE_CHUNK_TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_UPDATE_TITLE_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Chunk's propositions:
                {propositions}

                Chunk summary:
                {current_summary}

                Current chunk title:
                {current_title}
                """
            )
        ),
    ]
)
"""Prompt for updating an existing chunk's title after adding a proposition.

Template Variables:
    propositions: Newline-separated list of propositions in the chunk.
    current_summary: The chunk's current summary text.
    current_title: The chunk's current title.
"""


# System message for new chunk summary generation
_NEW_SUMMARY_SYSTEM = """You are the steward of a group of chunks which
represent groups of sentences that talk about a similar topic.
You should generate a very brief 1-sentence summary which will inform
viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying
instructions on what to add to the chunk.

You will be given a proposition which will go into a new chunk. This new
chunk needs a summary.

Your summaries should anticipate generalization. If you get a proposition
about apples, generalize it to food. Or month, generalize it to
"date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes
to eat.

Only respond with the new chunk summary, nothing else."""

NEW_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_NEW_SUMMARY_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Determine the summary of the new chunk that this proposition
                will go into:
                {proposition}
                """
            )
        ),
    ]
)
"""Prompt for generating an initial summary for a new chunk.

Template Variables:
    proposition: The first proposition being added to the new chunk.
"""


# System message for new chunk title generation
_NEW_TITLE_SYSTEM = """You are the steward of a group of chunks which
represent groups of sentences that talk about a similar topic.
You should generate a very brief few word chunk title which will inform
viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about.

You will be given a summary of a chunk which needs a title.

Your titles should anticipate generalization. If you get a proposition about
apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else."""

NEW_CHUNK_TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_NEW_TITLE_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Determine the title of the chunk that this summary belongs to:
                {summary}
                """
            )
        ),
    ]
)
"""Prompt for generating an initial title for a new chunk.

Template Variables:
    summary: The summary text of the new chunk.
"""


# System message for finding relevant chunks
_FIND_CHUNK_SYSTEM = """Determine whether or not the "Proposition" should
belong to any of the existing chunks.

A proposition should belong to a chunk if their meaning, direction, or
intention are similar. The goal is to group similar propositions and chunks.

If you think a proposition should be joined with a chunk, return the chunk id.
If you do not think an item should be joined with an existing chunk, just
return "No chunks".

Example:
Input:
    - Proposition: "Greg really likes hamburgers"
    - Current Chunks:
        - Chunk ID: 2n4l3d
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833k
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833k"""

FIND_RELEVANT_CHUNK_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_FIND_CHUNK_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Current Chunks:
                --Start of current chunks--
                {current_chunk_outline}
                --End of current chunks--
                """
            )
        ),
        HumanMessagePromptTemplate.from_template(
            textwrap.dedent(
                """
                Determine if the following statement should belong to one of
                the chunks outlined:
                {proposition}
                """
            )
        ),
    ]
)
"""Prompt for finding which existing chunk a proposition should belong to.

Template Variables:
    current_chunk_outline: Formatted string of all current chunks with their
        IDs, titles, and summaries.
    proposition: The proposition to find a matching chunk for.
"""
