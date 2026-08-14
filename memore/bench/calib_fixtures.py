"""The authored half of the calibration query set (RESULTS.md §5).

The benchmark supplies real questions against a real corpus, but its corpus is
encyclopedic -- 455 pre-stated facts about sports, birthplaces and universities. The
gate will actually run against a *conversational* session holding a dozen facts about
one user, and the score distribution depends on the corpus, not only on the model. So
the calibration measures both regimes and only trusts a floor that survives both.

This module is the conversational one. It is authored, not sampled, and that is stated
plainly wherever the numbers are reported: there is no public corpus of "gateway turns
against a personal memory store". What keeps it honest is the negative set -- these are
the turns a real session is full of (chit-chat, coding questions, meta-requests), and
several sit deliberately close to the stored facts ("should I use tabs or spaces?"
against a store that holds an editor preference) so the floor is not chosen against
straw negatives.
"""

from __future__ import annotations

# (fact, subject_hint) -- the shape P1 would emit for a learn-turn. The first two are
# the recall-poc-spec.md §6 demo facts, so the trace that defines "it works" is inside
# the calibration set rather than checked separately.
CHAT_FACTS: list[tuple[str, str]] = [
    ("deploys to staging by default", "deploy target"),
    ("prefers Python for backend work", "preferred backend language"),
    ("is based in Amsterdam", "location"),
    ("uses Neovim as their editor", "editor"),
    ("runs tests with pytest", "test runner"),
    ("has a dog named Pixel", "dog name"),
    ("works at a company called Northgate", "employer"),
    ("prefers dark mode in the terminal", "terminal theme preference"),
    ("is allergic to peanuts", "allergies"),
    ("drinks oat milk in coffee", "coffee preference"),
    ("manages a team of six engineers", "team size"),
    ("flies out to Lisbon on the 14th", "upcoming travel"),
]

# (query, index into CHAT_FACTS the gate should surface). Deliberately phrased as turns
# rather than as the fact restated -- a query that echoes the fact's wording measures
# string overlap, not recall.
CHAT_POSITIVES: list[tuple[str, int]] = [
    ("what's my deploy setup?", 0),
    ("where do I deploy again?", 0),
    ("what's my default deploy target?", 0),
    ("which language should I use for the backend?", 1),
    ("remind me which language I prefer", 1),
    ("what city am I in?", 2),
    ("where do I live?", 2),
    ("which editor do I use?", 3),
    ("what's my editor setup?", 3),
    ("how do I run the tests?", 4),
    ("which test framework do I run?", 4),
    # recall-poc-spec.md §6 uses this turn to show the *write* path storing nothing
    # (transient, no durable fact). That says nothing about the read path: the store does
    # hold the test runner, and surfacing it here is a correct recall, not a false open.
    # It was labelled a negative in the first calibration run and was the single largest
    # false-open for all three models -- a mislabel, corrected here rather than left to
    # flatter the numbers.
    ("run the tests again", 4),
    ("what's my dog called?", 5),
    ("tell me about my dog", 5),
    ("who do I work for?", 6),
    ("what company do I work at?", 6),
    ("do I prefer light or dark mode?", 7),
    ("any food allergies I should know about?", 8),
    ("am I allergic to anything?", 8),
    ("what do I take in my coffee?", 9),
    ("what kind of milk do I use?", 9),
    ("how big is my team?", 10),
    ("how many engineers report to me?", 10),
    ("when am I flying to Lisbon?", 11),
    ("what are my travel plans?", 11),
]

# Turns where the gate must stay shut: the store has nothing that answers them. Mixed
# deliberately -- pure off-domain chit-chat, plus a tail of near-misses that share the
# store's register ("what's on my calendar today?", "should I use tabs or spaces?").
OFF_DOMAIN_NEGATIVES: list[str] = [
    "what's the weather in Paris?",
    "how do I center a div in CSS?",
    "explain the difference between TCP and UDP",
    "write me a haiku about autumn",
    "what time is it in Tokyo?",
    "summarize this article for me",
    "can you explain that again?",
    "thanks, that helps",
    "what's 17 times 23?",
    "translate 'good morning' into Japanese",
    "give me a recipe for banana bread",
    "what does HTTP 429 mean?",
    "how long should I boil an egg?",
    "draft an email declining a meeting",
    "explain recursion like I'm five",
    "is it going to rain tomorrow?",
    "set a timer for 10 minutes",
    "help me debug this stack trace",
    "how do I revert a git commit?",
    "what are the side effects of ibuprofen?",
    "recommend a sci-fi novel",
    "convert 200 pounds to kilograms",
    "what's the difference between a list and a tuple?",
    "how many calories are in an avocado?",
    "explain quantum entanglement briefly",
    "why is the sky blue?",
    "what's a good name for a startup?",
    "tell me a joke",
    "what does this error mean: ECONNREFUSED",
    "how far away is the moon?",
    "write a SQL query joining two tables",
    "open the file and show me line 40",
    "what's on my calendar today?",
    "should I use tabs or spaces?",
    "which of these two designs looks better?",
    "book a table for two at 8pm",
    "remind me to call the dentist",
    "what's the best way to learn guitar?",
    "can you rewrite that paragraph more concisely?",
]


# --- a conversational session whose subjects COMPETE --------------------------------
#
# `CHAT_FACTS` above holds one fact per subject with no two subjects sharing a relation,
# which is a real shape (a short session) but a lucky one: the wrong-subject failure
# cannot occur there, so it cannot measure the defence against it either. A session that
# runs for any length of time accumulates competing subjects -- two services with deploy
# targets, two teams with leads, two pets with breeds -- and that is exactly the shape
# where similarity alone starts answering questions about the wrong entity.
#
# This fixture is that shape, and unlike `CHAT_FACTS` it carries HARD negatives: queries
# in a relation the store holds, about an entity it does not.
CROWDED_CHAT_FACTS: list[tuple[str, str]] = [
    ("the web app deploys to staging", "deploy target of the web app"),
    ("the api deploys to prod", "deploy target of the api"),
    ("the docs site deploys to netlify", "deploy target of the docs site"),
    ("the work laptop runs Linux", "operating system of the work laptop"),
    ("the home desktop runs Windows", "operating system of the home desktop"),
    ("Pixel is a border collie", "breed of Pixel"),
    ("Mochi is a siamese cat", "breed of Mochi"),
    ("Sarah leads the platform team", "lead of the platform team"),
    ("Tom leads the data team", "lead of the data team"),
    ("the platform team has six engineers", "size of the platform team"),
    ("the data team has four engineers", "size of the data team"),
    ("the staging cluster is in eu-west-1", "region of the staging cluster"),
    ("the prod cluster is in us-east-1", "region of the prod cluster"),
]

CROWDED_CHAT_POSITIVES: list[tuple[str, int]] = [
    ("where does the web app deploy?", 0),
    ("what's the deploy target for the api?", 1),
    ("where does the docs site go?", 2),
    ("what OS is on the work laptop?", 3),
    ("what does the home desktop run?", 4),
    ("what breed is Pixel?", 5),
    ("what kind of cat is Mochi?", 6),
    ("who leads the platform team?", 7),
    ("who runs the data team?", 8),
    ("how big is the platform team?", 9),
    ("how many engineers are on the data team?", 10),
    ("which region is the staging cluster in?", 11),
    ("where is the prod cluster hosted?", 12),
]

# Same relations, entities the store has never heard of. Under similarity alone these
# match a sibling fact almost perfectly.
CROWDED_CHAT_HARD_NEGATIVES: list[str] = [
    "where does the mobile app deploy?",
    "what's the deploy target for the admin panel?",
    "where does the scheduler deploy?",
    "what OS does the media server run?",
    "what does the tablet run?",
    "what breed is Luna?",
    "what kind of dog is Biscuit?",
    "who leads the design team?",
    "who runs the QA team?",
    "how big is the security team?",
    "how many engineers are on the mobile team?",
    "which region is the dev cluster in?",
    "where is the analytics cluster hosted?",
]
