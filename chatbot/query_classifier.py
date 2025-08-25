import spacy
import re

class QueryClassifier:
    def __init__(self, llm_pipe=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.llm_pipe = llm_pipe
        self.geo_triggers = {
            "explicit": ["map", "europe", "coast", "latitude", "longitude", "geopandas", "shapefile", "cluster on map"],
            "operations": ["plot points", "markers on", "spatial distribution", "on a map", "heatmap"],
            "implicit": ["near the coast", "across regions", "geographic distribution"]
        }
        self.non_geo_triggers = {
            "statistical": ["histogram", "scatterplot", "regression", "distribution plot", "correlation matrix"],
            "analysis": ["significant", "p-value", "confidence interval", "hypothesis test", "anova", "t-test"]
        }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        
        # Stage 1: Direct geo triggers
        for term in self.geo_triggers["explicit"] + self.geo_triggers["operations"]:
            if term in query_lower:
                print("Geo triggers ACTIVATED")
                return "geo"
        
        # Stage 2: Statistical/inferential triggers
        for term in self.non_geo_triggers["statistical"] + self.non_geo_triggers["analysis"]:
            if term in query_lower:
                print("Non-geo triggers ACTIVATED")
                return "infer" if any(t in query_lower for t in ["test", "signif"]) else "desc"
        
        # Stage 3: Syntactic analysis
        if self._has_geo_context(query):
            print("GEO on Syntactic analysis ACTIVATED")
            return "geo"
        
        # Stage 4: LLM fallback
        print("GEO on LLM ACTIVATED")
        return self._llm_classification(query)

    def _has_geo_context(self, query: str) -> bool:
        doc = self.nlp(query)
        geo_score = 0
        
        # Check for spatial prepositions
        for token in doc:
            if token.text.lower() in ["near", "across"]: # Mislim, postoje sigurno ti neki izrazi kao in i za statistike
                for child in token.children:
                    if child.ent_type_ in ["GPE", "LOC"]:
                        geo_score += 2
        
        # Check for coordinate references
        if re.search(r"\b(lat|long|coordinates?)\b", query, re.I):
            geo_score += 2
            
        return geo_score >= 2

    def _llm_classification(self, query: str) -> str:
        if not self.llm_pipe: # I don't know what this means
            return "desc"  # Fallback
            
        prompt = f"""Classify this agricultural query into [geo], [desc], or [infer]:
        Rules:
        - geo: Requires maps/coordinates/shapefiles
        - desc: Basic statistics/data visualization
        - infer: Statistical tests/models
        
        Query: "{query}"
        Answer ONLY with one word in lowercase."""
        
        result = self.llm_pipe(prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]
        print(result)
        return re.search(r"\b(geo|desc|infer)\b", result.lower()).group(0) if result else "desc"