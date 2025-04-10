class DatasetColumns:
    SOURCE_PROMPT = 'Source Prompt'
    TARGET_PROMPT = 'Target Prompt'
    SOURCE_DOMAIN = 'Source Domain'
    TARGET_DOMAIN = 'Target Domain'
    PAIR_TYPE = 'Pair Type'
    LABELER = 'Labeler'
    LEVEL = 'Level'
    SOURCE_ID = 'Source ID'
    TARGET_ID = 'Target ID'
    SOURCE_MODEL = 'Source Model'
    TARGET_MODEL = 'Target Model'
    SOURCE_TEXT = 'Source Text'
    TARGET_TEXT = 'Target Text'
    SOURCE_NUMBERED_TEXT = 'Source Numbered Text'
    TARGET_NUMBERED_TEXT = 'Target Numbered Text'
    SOURCE_NUMBER_TO_SENTENCE = 'Source Sentence_Number_Dict'
    TARGET_NUMBER_TO_SENTENCE = 'Target Sentence_Number_Dict'
    SOURCE_SEGMENTS = 'Source Segments'
    TARGET_SEGMENTS = 'Target Segments'
    SOURCE_QUDS = 'Source QUDs'
    TARGET_QUDS = 'Target QUDs'
    SOURCE_QUD_TO_SEGMENT = 'Source QUD to Segment Dict'
    TARGET_QUD_TO_SEGMENT = 'Target QUD to Segment Dict'
    SOURCE_SEGMENT_TO_QUD = 'Source Segment to QUD Dict'
    TARGET_SEGMENT_TO_QUD = 'Target Segment to QUD Dict'
    QUD_ANSWERS = 'QUD Answers'
    QUD_ANSWER_SEGMENTS = 'QUD Answer Segments'

class Models:
    GEMINI_MODEL = "gemini-1.5-flash-002"
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    GPT_MODEL = "gpt-4o-2024-08-06" 
    HUMAN = "Human"

class Domain:
    CREATIVE_WRITING = "Fiction"
    OBITUARY = "Obituary"
    SURI = "SURI"

class Labelers:
    ANNOTATOR = "human-annotation"
    MODEL = Models.GPT_MODEL
    HUMAN_GOLD = "human-intersection"

class PairType:
    IN_DOMAIN = "in-domain"
    MINIMAL_PAIR = "minimal-pair"

class PairColumns:
    ID1 = "Doc1 ID"
    ID2 = "Doc2 ID"
    LEVEL = 'Level'
    MODEL1 = 'Model 1'
    MODEL2 = 'Model 2'
    DOC1 = 'Document 1'
    DOC2 = 'Document 2'
    NUMBERED_TEXT1 = 'Doc1 Numbered Text'
    NUMBERED_TEXT2 = 'Doc2 Numbered Text'
    NUMBER_TO_SENTENCE1 = 'Doc1 Sentence_Number_Dict'
    NUMBER_TO_SENTENCE2 = 'Doc2 Sentence_Number_Dict'
    SEGMENTS1 = 'Doc1 Segments'
    SEGMENTS2 = 'Doc2 Segments'
    ABSTRACTED_ENTITIES1 = 'Doc1 Abstracted Entities'
    ABSTRACTED_ENTITIES2 = 'Doc2 Abstracted Entities'
    QUDS1 = 'Doc1 QUDs'
    QUDS2 = 'Doc2 QUDs'
    QUD_TO_SEGMENT1 = 'Doc1 QUD to Segment Dict'
    QUD_TO_SEGMENT2 = 'Doc2 QUD to Segment Dict'
    SEGMENT_TO_QUD1 = 'Doc1 Segment to QUD Dict'
    SEGMENT_TO_QUD2 = 'Doc2 Segment to QUD Dict'
    D1_TO_D2_QUD_ANSWERS = 'D1 to D2 QUD Answers'
    # D1_TO_D2_ANSWER_SEGMENTS = 'D1 to D2 Answer Segments'
    D2_TO_D1_QUD_ANSWERS = 'D2 to D1 QUD Answers'
    # D2_TO_D1_ANSWER_SEGMENTS = 'D2 to D1 Answer Segments'
    HARMONIC_SCORE = 'Harmonic QUDsim Score'
    ALIGNED_SEGMENTS = 'QUDsim Aligned Segment Indices'
    ALIGNED_SEGMENT_TEXT = 'QUDsim Aligned Segments'

REQUIRED_FIELDS = [PairColumns.DOC1, PairColumns.DOC2, PairColumns.MODEL1, PairColumns.MODEL2]

class DocumentColumns:
    ID = 'ID'
    LEVEL = 'Level'
    MODEL = 'Model'
    DOCUMENT = 'Text'
    NUMBERED_TEXT = 'Numbered Text'
    NUMBER_TO_SENTENCE = 'Sentence_Number_Dict'
    SEGMENTS = 'Segments'
    ABSTRACTED_ENTITIES = 'Abstracted Entities'
    QUDS = 'QUDs'
    QUD_TO_SEGMENT = 'QUD to Segment Dict'
    SEGMENT_TO_QUD = 'Segment to QUD Dict'