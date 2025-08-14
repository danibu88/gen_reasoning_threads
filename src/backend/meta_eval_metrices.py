import re
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))


class FinalFourMetrics:
    def __init__(self):
        """Initialize the four key metrics calculator"""
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Domain-specific term dictionaries
        self.domain_terms = {
            "healthcare": ["patient", "medical", "clinical", "diagnosis", "treatment", "therapy",
                           "hospital", "doctor", "nurse", "medication", "prescription", "healthcare",
                           "disease", "symptom", "vital", "health", "care", "medical record"],
            "business": ["customer", "market", "revenue", "profit", "sales", "marketing",
                         "strategy", "business", "company", "enterprise", "stakeholder", "roi",
                         "kpi", "budget", "investment", "growth", "competition", "brand"],
            "technology": ["api", "database", "system", "software", "application", "platform",
                           "server", "cloud", "network", "security", "algorithm", "code",
                           "programming", "development", "tech", "digital", "computing"],
            "finance": ["financial", "investment", "portfolio", "trading", "banking", "credit",
                        "loan", "interest", "capital", "asset", "liability", "risk", "return",
                        "market", "stock", "bond", "funds", "money"],
            "education": ["student", "teacher", "learning", "education", "curriculum", "course",
                          "training", "knowledge", "skill", "academic", "school", "university",
                          "assessment", "grade", "study", "research"],
            "manufacturing": ["production", "manufacturing", "assembly", "quality", "process",
                              "factory", "equipment", "supply chain", "inventory", "logistics",
                              "operations", "efficiency", "automation", "maintenance"]
        }

        # Technical implementation terms
        self.technical_terms = [
            # Programming and development
            "implement", "develop", "code", "program", "script", "function", "class", "method",
            "library", "framework", "module", "package", "import", "install", "configure",

            # Systems and infrastructure
            "server", "database", "api", "endpoint", "url", "http", "https", "rest", "json",
            "xml", "sql", "nosql", "mongodb", "mysql", "postgresql", "redis", "cache",

            # Cloud and deployment
            "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "container", "deploy",
            "deployment", "ci/cd", "devops", "infrastructure", "microservices", "serverless",

            # Data and AI/ML
            "algorithm", "model", "machine learning", "ai", "artificial intelligence", "neural",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "data", "dataset",

            # Specific technologies
            "python", "javascript", "java", "c++", "react", "angular", "vue", "node.js",
            "spring", "django", "flask", "fastapi", "git", "github", "gitlab", "jenkins"
        ]

        # Sequential indicators for coherence
        self.sequence_indicators = [
            # Step indicators
            r"step\s+\d+", r"instruction\s+\d+", r"\d+\.", r"\d+\)",

            # Temporal indicators
            "first", "second", "third", "then", "next", "after", "before", "finally",
            "initially", "subsequently", "meanwhile", "once", "when", "while",

            # Logical connectors
            "therefore", "thus", "hence", "consequently", "as a result", "because",
            "since", "given that", "provided that", "assuming", "if", "unless",

            # Continuation indicators
            "furthermore", "moreover", "additionally", "also", "in addition",
            "building on", "expanding", "extending"
        ]

    def calculate_actionability(self, instructions, user_prompt):
        """
        Measure how easy instructions are to implement based on closeness to user prompt

        Score components:
        1. Semantic similarity to original prompt (40%)
        2. Presence of clear action verbs (25%)
        3. Specificity of actions (not vague) (20%)
        4. Practical implementability (15%)
        """
        if not instructions or not user_prompt:
            return 0.0

        # 1. Semantic similarity to user prompt
        prompt_embedding = self.sentence_model.encode(user_prompt, convert_to_tensor=True)
        instructions_embedding = self.sentence_model.encode(instructions, convert_to_tensor=True)
        semantic_similarity = util.cos_sim(prompt_embedding, instructions_embedding).item()

        # 2. Action verbs presence
        action_verbs = [
            "implement", "create", "develop", "build", "design", "configure", "setup", "install",
            "deploy", "integrate", "connect", "establish", "enable", "activate", "initialize",
            "execute", "run", "start", "launch", "test", "validate", "verify", "monitor"
        ]

        instructions_lower = instructions.lower()
        action_verb_count = sum(1 for verb in action_verbs if verb in instructions_lower)
        action_verb_score = min(action_verb_count / 5, 1.0)  # Normalize to max 1.0

        # 3. Specificity (avoid vague terms)
        vague_terms = ["somehow", "maybe", "probably", "might", "could", "should consider",
                       "try to", "attempt to", "think about", "look into", "explore"]
        vague_count = sum(1 for term in vague_terms if term in instructions_lower)
        specificity_score = max(0, 1.0 - (vague_count / 3))  # Penalize vagueness

        # 4. Practical implementability (presence of concrete tools/methods)
        concrete_terms = ["using", "with", "via", "through", "by implementing", "by configuring",
                          "tool", "library", "framework", "platform", "service", "software"]
        concrete_count = sum(1 for term in concrete_terms if term in instructions_lower)
        practical_score = min(concrete_count / 3, 1.0)

        # Weighted combination
        actionability_score = (
                semantic_similarity * 0.40 +
                action_verb_score * 0.25 +
                specificity_score * 0.20 +
                practical_score * 0.15
        )

        return round(actionability_score, 4)

    def calculate_coherence(self, instructions):
        """
        Measure if instructions are in order and can be sequentially implemented

        Score components:
        1. Presence of sequential indicators (30%)
        2. Logical dependency structure (25%)
        3. Clear step numbering/ordering (25%)
        4. Consistent flow between sentences (20%)
        """
        if not instructions:
            return 0.0

        instructions_lower = instructions.lower()
        sentences = sent_tokenize(instructions)

        # 1. Sequential indicators
        sequence_count = 0
        for pattern in self.sequence_indicators:
            if isinstance(pattern, str):
                sequence_count += instructions_lower.count(pattern)
            else:  # regex pattern
                sequence_count += len(re.findall(pattern, instructions_lower))

        sequence_score = min(sequence_count / len(sentences), 1.0)

        # 2. Logical dependency structure
        dependency_words = ["after", "before", "once", "when", "then", "next", "following"]
        dependency_count = sum(1 for word in dependency_words if word in instructions_lower)
        dependency_score = min(dependency_count / 3, 1.0)

        # 3. Clear step numbering/ordering
        step_patterns = [
            r"step\s+\d+", r"instruction\s+\d+", r"\d+\.", r"\d+\)",
            r"first", r"second", r"third", r"fourth", r"fifth"
        ]
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, instructions_lower))

        step_score = min(step_count / 3, 1.0)  # Expect at least 3 steps

        # 4. Sentence flow consistency
        if len(sentences) > 1:
            sentence_embeddings = [self.sentence_model.encode(sent, convert_to_tensor=True) for sent in sentences]
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i + 1]).item()
                similarities.append(sim)

            # High similarity indicates good flow, but not too high (redundancy)
            avg_similarity = np.mean(similarities)
            # Optimal range: 0.3 to 0.7
            if 0.3 <= avg_similarity <= 0.7:
                flow_score = 1.0
            elif avg_similarity < 0.3:
                flow_score = avg_similarity / 0.3
            else:  # > 0.7
                flow_score = max(0, 1.0 - (avg_similarity - 0.7) / 0.3)
        else:
            flow_score = 0.5  # Single sentence gets neutral score

        # Weighted combination
        coherence_score = (
                sequence_score * 0.30 +
                dependency_score * 0.25 +
                step_score * 0.25 +
                flow_score * 0.20
        )

        return round(coherence_score, 4)

    def calculate_domain_specificity(self, instructions, domain_hint=None):
        """
        Measure if instructions are specific to the domain of the prompt

        Score components:
        1. Domain-specific terminology frequency (50%)
        2. Domain context relevance (30%)
        3. Generic vs specific language ratio (20%)
        """
        if not instructions:
            return 0.0

        instructions_lower = instructions.lower()

        # 1. Domain-specific terminology
        domain_scores = {}
        total_words = len(instructions_lower.split())

        for domain, terms in self.domain_terms.items():
            term_count = sum(1 for term in terms if term in instructions_lower)
            domain_scores[domain] = term_count / max(total_words, 1)

        # Find dominant domain
        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]

        # If domain_hint provided, use it; otherwise use detected domain
        if domain_hint and domain_hint.lower() in domain_scores:
            domain_score = domain_scores[domain_hint.lower()]
        else:
            domain_score = best_score

        # Normalize domain score
        domain_terminology_score = min(domain_score * 100, 1.0)  # Scale up

        # 2. Domain context relevance (semantic similarity to domain)
        if domain_hint and domain_hint.lower() in self.domain_terms:
            domain_context = " ".join(self.domain_terms[domain_hint.lower()])
        else:
            domain_context = " ".join(self.domain_terms[best_domain])

        domain_embedding = self.sentence_model.encode(domain_context, convert_to_tensor=True)
        instructions_embedding = self.sentence_model.encode(instructions, convert_to_tensor=True)
        context_similarity = util.cos_sim(domain_embedding, instructions_embedding).item()

        # 3. Generic vs specific language
        generic_terms = ["system", "solution", "approach", "method", "way", "process",
                         "thing", "item", "element", "component", "part", "aspect"]
        generic_count = sum(1 for term in generic_terms if term in instructions_lower)
        generic_penalty = min(generic_count / 5, 0.5)  # Max 50% penalty
        specificity_score = max(0, 1.0 - generic_penalty)

        # Weighted combination
        domain_specificity_score = (
                domain_terminology_score * 0.50 +
                context_similarity * 0.30 +
                specificity_score * 0.20
        )

        return round(domain_specificity_score, 4)

    def calculate_technological_specificity(self, instructions):
        """
        Measure if instructions are directly implementable with technical concepts

        Score components:
        1. Technical terminology frequency (40%)
        2. Specific tools/technologies mentioned (30%)
        3. Implementation details provided (20%)
        4. Code/configuration examples or references (10%)
        """
        if not instructions:
            return 0.0

        instructions_lower = instructions.lower()
        total_words = len(instructions_lower.split())

        # 1. Technical terminology frequency
        tech_term_count = sum(1 for term in self.technical_terms if term in instructions_lower)
        tech_frequency_score = min(tech_term_count / 5, 1.0)  # Normalize

        # 2. Specific tools/technologies mentioned
        specific_tools = [
            # Programming languages
            "python", "javascript", "java", "c++", "c#", "go", "rust", "php", "ruby",

            # Frameworks and libraries
            "react", "angular", "vue", "django", "flask", "spring", "tensorflow", "pytorch",
            "pandas", "numpy", "opencv", "keras", "scikit-learn",

            # Databases and storage
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",

            # Cloud platforms and services
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "github", "gitlab",

            # APIs and protocols
            "rest", "graphql", "soap", "http", "https", "json", "xml", "oauth",

            # Specific products
            "salesforce", "shopify", "wordpress", "stripe", "twilio", "slack", "zoom"
        ]

        specific_tool_count = sum(1 for tool in specific_tools if tool in instructions_lower)
        tool_specificity_score = min(specific_tool_count / 3, 1.0)

        # 3. Implementation details
        implementation_indicators = [
            "configure", "setup", "install", "initialize", "authenticate", "connect to",
            "endpoint", "url", "port", "parameter", "variable", "function", "method",
            "class", "object", "instance", "callback", "webhook", "token", "key"
        ]

        implementation_count = sum(1 for indicator in implementation_indicators
                                   if indicator in instructions_lower)
        implementation_score = min(implementation_count / 4, 1.0)

        # 4. Code/configuration references
        code_indicators = [
            "code", "script", "config", "configuration", "yaml", "json", "xml",
            "environment variable", "command line", "terminal", "bash", "shell",
            "import", "require", "include", "library", "package", "dependency"
        ]

        code_count = sum(1 for indicator in code_indicators if indicator in instructions_lower)
        code_score = min(code_count / 2, 1.0)

        # Weighted combination
        technological_specificity_score = (
                tech_frequency_score * 0.40 +
                tool_specificity_score * 0.30 +
                implementation_score * 0.20 +
                code_score * 0.10
        )

        return round(technological_specificity_score, 4)

    def calculate_all_four_metrics(self, instructions, user_prompt, domain_hint=None):
        """
        Calculate all four key metrics for given instructions

        Returns:
            dict: Dictionary with all four metric scores
        """
        return {
            "actionability": self.calculate_actionability(instructions, user_prompt),
            "coherence": self.calculate_coherence(instructions),
            "domain_specificity": self.calculate_domain_specificity(instructions, domain_hint),
            "technological_specificity": self.calculate_technological_specificity(instructions)
        }

    # Modified version of your existing function to include enhanced metrics


def calculate_enhanced_understandability(instructions):
    """Enhanced understandability metric (0-10 scale)"""
    if not instructions or len(instructions.strip()) < 10:
        return 0.0

    try:
        score = 0.0

        # 1. Structural Clarity (0-3 points)
        lines = instructions.split('\n')
        structure_indicators = 0

        # Check for numbered/bulleted lists
        if any(re.match(r'^\s*[\d\-\*\+•]', line) for line in lines):
            structure_indicators += 1

        # Check for headers or sections
        if any(re.match(r'^#+\s', line) or (line.isupper() and len(line) < 50)
               for line in lines if line.strip()):
            structure_indicators += 1

        # Check for paragraph separation
        if instructions.count('\n\n') >= 1:
            structure_indicators += 1

        # Check for reasonable line lengths
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            avg_length = np.mean([len(line) for line in non_empty_lines])
            if 20 <= avg_length <= 100:
                structure_indicators += 1

        structure_score = structure_indicators / 4
        score += structure_score * 3

        # 2. Readability (0-2.5 points)
        try:
            sentences = sent_tokenize(instructions)
            words = word_tokenize(instructions)

            if sentences and words:
                # Average sentence length (shorter is more readable)
                avg_sentence_length = len(words) / len(sentences)
                length_score = max(0, 1 - (avg_sentence_length - 15) / 30)

                # Complex words ratio
                complex_words = sum(1 for word in words if len(word) > 6 and word.isalpha())
                complexity_ratio = complex_words / len(words) if words else 0
                complexity_score = max(0, 1 - complexity_ratio * 2)

                readability_score = (length_score + complexity_score) / 2
                score += readability_score * 2.5
        except:
            score += 1.25  # Default moderate score

        # 3. Step Clarity (0-2.5 points)
        lines = [line.strip() for line in instructions.split('\n') if line.strip()]

        if lines:
            clarity_scores = []

            for line in lines:
                line_score = 0

                # Starts with action verb
                action_verbs = ['create', 'build', 'install', 'configure', 'set', 'run',
                                'execute', 'open', 'click', 'enter', 'select', 'save']
                if any(line.lower().startswith(verb) for verb in action_verbs):
                    line_score += 0.25

                # Contains specific details
                if re.search(r'\b(?:in|on|at|using|with|from|to)\s+\w+', line.lower()):
                    line_score += 0.25

                # Reasonable length
                if 15 <= len(line) <= 120:
                    line_score += 0.25

                # Contains concrete nouns
                concrete_nouns = ['file', 'folder', 'button', 'menu', 'screen', 'field', 'option']
                if any(noun in line.lower() for noun in concrete_nouns):
                    line_score += 0.25

                clarity_scores.append(line_score)

            step_clarity_score = np.mean(clarity_scores)
            score += step_clarity_score * 2.5

        # 4. Cognitive Load (0-2 points) - inverted scale
        try:
            words = word_tokenize(instructions)
            sentences = sent_tokenize(instructions)

            if words and sentences:
                load_factors = []

                # Sentence complexity
                avg_words_per_sentence = len(words) / len(sentences)
                sentence_load = min(1.0, avg_words_per_sentence / 25)
                load_factors.append(sentence_load)

                # Technical jargon density
                tech_jargon = ['algorithm', 'framework', 'architecture', 'implementation',
                               'configuration', 'optimization', 'paradigm', 'methodology']
                jargon_count = sum(1 for word in words if word.lower() in tech_jargon)
                jargon_load = min(1.0, (jargon_count / len(words)) * 20)
                load_factors.append(jargon_load)

                avg_load = np.mean(load_factors)
                cognitive_load_score = 1.0 - avg_load  # Invert for understandability
                score += cognitive_load_score * 2
        except:
            score += 1  # Default moderate score

        return min(10.0, max(0.0, score))

    except Exception as e:
        print(f"Error calculating enhanced understandability: {e}")
        return 5.0


def calculate_enhanced_user_focus(prompt, instructions, user_entities=None):
    """Enhanced user focus that measures closeness to prompt (0-10 scale)"""
    if not instructions or not prompt:
        return 0.0

    try:
        score = 0.0

        # 1. Basic Entity Coverage (0-4 points) - your existing approach
        if user_entities:
            entity_score = 0
            instructions_lower = instructions.lower()
            for entity in user_entities:
                if entity and entity.lower() in instructions_lower:
                    entity_score += 1
            entity_coverage = min(4.0, entity_score * 0.8)
            score += entity_coverage
        else:
            score += 2.0  # Default if no entities provided

        # 2. Intent Preservation (0-3 points)
        intent_patterns = [
            r'\b(create|build|develop|implement|design|setup|configure)\b',
            r'\b(analyze|evaluate|assess|review|examine)\b',
            r'\b(integrate|connect|link|combine|merge)\b',
            r'\b(optimize|improve|enhance|upgrade)\b',
            r'\b(automate|streamline|simplify)\b'
        ]

        prompt_intents = set()
        instruction_intents = set()

        for pattern in intent_patterns:
            prompt_matches = re.findall(pattern, prompt.lower())
            instruction_matches = re.findall(pattern, instructions.lower())
            prompt_intents.update(prompt_matches)
            instruction_intents.update(instruction_matches)

        if prompt_intents:
            preserved = prompt_intents.intersection(instruction_intents)
            intent_score = len(preserved) / len(prompt_intents)
        else:
            intent_score = 1.0  # No intents to preserve

        score += intent_score * 3

        # 3. Keyword Overlap (0-2 points)
        try:
            prompt_words = set(word.lower() for word in word_tokenize(prompt)
                               if word.lower() not in STOP_WORDS and len(word) > 2 and word.isalnum())
            instruction_words = set(word.lower() for word in word_tokenize(instructions)
                                    if word.lower() not in STOP_WORDS and len(word) > 2 and word.isalnum())

            if prompt_words:
                overlap = len(prompt_words.intersection(instruction_words))
                keyword_score = overlap / len(prompt_words)
            else:
                keyword_score = 1.0
        except:
            keyword_score = 0.5

        score += keyword_score * 2

        # 4. Requirement Coverage (0-1 point)
        requirement_indicators = ['should', 'must', 'need', 'require', 'want', 'expect']

        prompt_requirements = []
        for indicator in requirement_indicators:
            pattern = f"{indicator}\\s+(\\w+(?:\\s+\\w+)?)"
            matches = re.findall(pattern, prompt.lower())
            prompt_requirements.extend(matches)

        if prompt_requirements:
            covered = sum(1 for req in prompt_requirements
                          if req.lower() in instructions.lower())
            requirement_score = covered / len(prompt_requirements)
        else:
            requirement_score = 1.0

        score += requirement_score * 1

        return min(10.0, max(0.0, score))

    except Exception as e:
        print(f"Error calculating enhanced user focus: {e}")
        return 5.0


def calculate_final_four_metrics_enhanced(prompt, gnn_instructions, mcts_instructions, rmodel_instructions, domain):
    """
    Enhanced version of your existing function that includes understandability and user focus
    """
    # Call your existing function first
    existing_metrics = calculate_final_four_metrics(prompt, gnn_instructions, mcts_instructions,
                                                    rmodel_instructions, domain)

    # Add enhanced metrics
    enhanced_metrics = {}

    # Extract user entities from prompt (simple approach)
    user_entities = extract_user_entities_simple(prompt)

    instructions_dict = {
        'gnn': gnn_instructions,
        'mcts': mcts_instructions,
        'rmodel': rmodel_instructions
    }

    for model, instructions in instructions_dict.items():
        # Enhanced understandability
        enhanced_metrics[f"{model}_understandability_enhanced"] = calculate_enhanced_understandability(instructions)

        # Enhanced user focus (closeness to prompt)
        enhanced_metrics[f"{model}_user_focus_enhanced"] = calculate_enhanced_user_focus(prompt, instructions,
                                                                                         user_entities)

        # ADD THESE TWO LINES - Normalized versions for overall score calculation
        enhanced_metrics[f"{model}_understandability_normalized"] = enhanced_metrics[
                                                                        f"{model}_understandability_enhanced"] / 10.0
        enhanced_metrics[f"{model}_user_focus_normalized"] = enhanced_metrics[f"{model}_user_focus_enhanced"] / 10.0

    # Combine existing and enhanced metrics
    all_metrics = {**existing_metrics, **enhanced_metrics}

    return all_metrics


def extract_user_entities_simple(prompt):
    """Simple extraction of entities from prompt"""
    # Look for capitalized words and technical terms
    entities = []

    # Capitalized words (potential proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', prompt)
    entities.extend(capitalized)

    # Technical terms
    tech_terms = ['API', 'SDK', 'ML', 'AI', 'IoT', 'REST', 'JSON', 'XML', 'SQL', 'AWS', 'Azure', 'GCP']
    for term in tech_terms:
        if term.lower() in prompt.lower():
            entities.append(term)

    return list(set(entities))  # Remove duplicates


# Integration function for existing experiment scripts
def calculate_final_four_metrics(prompt, gnn_instructions, mcts_instructions, rmodel_instructions, domain_hint=None):
    """
    Calculate the four key metrics for all three models

    Args:
        prompt: User's original prompt
        gnn_instructions: Instructions from GNN model
        mcts_instructions: Instructions from MCTS model
        rmodel_instructions: Instructions from reasoning model
        domain_hint: Optional domain hint for better domain specificity calculation

    Returns:
        dict: Metrics for all three models
    """
    calculator = FinalFourMetrics()

    result = {}

    # Calculate for each model
    for model_name, instructions in [
        ("gnn", gnn_instructions),
        ("mcts", mcts_instructions),
        ("rmodel", rmodel_instructions)
    ]:
        if isinstance(instructions, dict):
            instructions = instructions.get("instructions", "")

        model_metrics = calculator.calculate_all_four_metrics(
            instructions, prompt, domain_hint
        )

        # Add model prefix to metrics
        for metric_name, score in model_metrics.items():
            result[f"{model_name}_{metric_name}"] = score

    return result


# Test function
def test_final_four_metrics():
    """Test the four metrics with sample data"""
    calculator = FinalFourMetrics()

    sample_prompt = "I need to build a healthcare system that tracks patient medications and sends alerts to family members"

    sample_instructions = {
        "good": """INSTRUCTION 1: Implement a patient medication tracking database using PostgreSQL with tables for patients, medications, and schedules.

INSTRUCTION 2: Configure automated alert system using Twilio API to send SMS notifications to registered family members when medications are missed.

INSTRUCTION 3: Deploy the healthcare monitoring dashboard using React frontend connected to Flask backend with authentication and patient data visualization.""",

        "poor": """INSTRUCTION 1: Think about creating some kind of system for tracking things.

INSTRUCTION 2: Maybe try to set up alerts somehow using some service.

INSTRUCTION 3: Build a solution that works for your needs."""
    }

    print("Testing Final Four Metrics")
    print("=" * 50)

    for quality, instructions in sample_instructions.items():
        print(f"\n{quality.upper()} Instructions:")
        metrics = calculator.calculate_all_four_metrics(instructions, sample_prompt, "healthcare")

        for metric, score in metrics.items():
            print(f"  {metric.title()}: {score:.3f}")


if __name__ == "__main__":
    test_final_four_metrics()
