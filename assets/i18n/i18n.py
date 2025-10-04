import os
import json
from pathlib import Path
from locale import getdefaultlocale

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

now_dir = os.path.dirname(os.path.abspath(__file__))

class I18nAuto:
    LANGUAGE_PATH = os.path.join(BASE_DIR, "assets", "i18n", "languages")

    def __init__(self, language=None):
        config_path = os.path.join(BASE_DIR, "assets", "config.json")
        try:
            with open(config_path, "r", encoding="utf8") as file:
                config = json.load(file)
                lang_config = config.get("lang", {"override": False, "selected_lang": "auto"})
                override = lang_config.get("override", False)
                lang_prefix = lang_config.get("selected_lang", "auto")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            override = False
            lang_prefix = "auto"

        self.language = lang_prefix

        if not override:
            language = language or getdefaultlocale()[0]
            lang_prefix = language[:2] if language is not None else "en"
            available_languages = self._get_available_languages()
            matching_languages = [lang for lang in available_languages if lang.startswith(lang_prefix)]
            self.language = matching_languages[0] if matching_languages else "en_US"

        self.language_map = self._load_language_list()

    def _load_language_list(self):
        try:
            file_path = Path(self.LANGUAGE_PATH) / f"{self.language}.json"
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load language file for {self.language}. Check if the correct .json file exists."
            )

    def _get_available_languages(self):
        language_files = [path.stem for path in Path(self.LANGUAGE_PATH).glob("*.json")]
        return language_files

    def _language_exists(self, language):
        return (Path(self.LANGUAGE_PATH) / f"{language}.json").exists()

    def __call__(self, key):
        return self.language_map.get(key, key)
