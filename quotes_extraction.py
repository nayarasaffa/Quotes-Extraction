import torch
import joblib
from transformers import XLMRobertaTokenizer
from model import EntityModel
from corpus import EntityDataset
from idsentsegmenter.sentence_segmentation import SentenceSegmentation

class QuoteEntityExtractor:
    def __init__(self, direct_model_path, indirect_model_path, tokenizer_name="xlm-roberta-base", cache_dir="models/transformers_cache"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)

        self.model_direct, self.enc_tag_direct, self.chars_direct = self._load_direct_model(direct_model_path)
        self.model_indirect, self.enc_tag_indirect, self.chars_indirect = self._load_indirect_model(indirect_model_path)

        self.prefer_indirect_labels = {
            "B-PERSON", "I-PERSON", "L-PERSON", "U-PERSON",
            "B-PERSONCOREF", "I-PERSONCOREF", "L-PERSONCOREF", "U-PERSONCOREF",
            "B-CUE", "I-CUE", "L-CUE", "U-CUE", 
            "B-CUECOREF", "I-CUECOREF", "L-CUECOREF", "U-CUECOREF",
            "B-STATEMENT", "I-STATEMENT", "L-STATEMENT", "U-STATEMENT"
        }

    def _load_direct_model(self, model_dir):
        meta = joblib.load(f"{model_dir}/meta_direct2.bin")
        enc_tag = meta["enc_tag"]
        chars = meta["chars"]

        model = EntityModel(num_tag=len(enc_tag.classes_), char_input_dim=len(chars))
        state_dict = torch.load(f"{model_dir}/direct-quotes.bin", map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model, enc_tag, chars

    def _load_indirect_model(self, model_dir):
        meta = joblib.load(f"{model_dir}/meta_indirect2.bin")
        enc_tag = meta["enc_tag"]
        chars = meta["chars"]

        model = EntityModel(num_tag=len(enc_tag.classes_), char_input_dim=len(chars))
        state_dict = torch.load(f"{model_dir}/indirect-quotes.bin", map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model, enc_tag, chars

    def predict_sentence(self, model, sentence, enc_tag, chars):
        tokens = sentence.split()
        dataset = EntityDataset([tokens], [[0] * len(tokens)], enc_tag=enc_tag, char_vocab=chars)
        data = dataset[0]
        for k in data:
            data[k] = data[k].to(self.device).unsqueeze(0)
        with torch.no_grad():
            tag_ids = model.encode(**data, device=self.device)[0]
        return enc_tag.inverse_transform(tag_ids)

    def reverse_tokenize(self, ids, tags):
        tokens = []
        tags_list = []
        prev_tag = None

        for token_id, tag in zip(ids, tags):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            if token in ['<s>', '</s>']:
                continue
            if token.startswith("▁"):
                token = token.replace("▁", "")
                tokens.append(token)
                tags_list.append(tag)
                prev_tag = tag
            else:
                if tokens:
                    tokens[-1] += token
                    tags_list[-1] = prev_tag
        return list(zip(tokens, tags_list))

    def convert_to_entities(self, token_tag_pairs, original_sentence):
        entities = []
        current_entity = ""
        current_label = None
        start_idx = None
        idx_in_text = 0

        for token, tag in token_tag_pairs:
            token_clean = token.strip("▁")
            while idx_in_text < len(original_sentence) and original_sentence[idx_in_text].isspace():
                idx_in_text += 1
            token_start = original_sentence.find(token_clean, idx_in_text)
            if token_start == -1:
                continue
            token_end = token_start + len(token_clean)

            if tag == "O":
                if current_entity and current_label:
                    entities.append({
                        "label": current_label,
                        "text": current_entity.strip(),
                        "start": start_idx,
                        "end": prev_token_end
                    })
                    current_entity = ""
                    current_label = None
                    start_idx = None
            else:
                tag_prefix, tag_type = tag.split("-", 1)
                if current_label == tag_type:
                    current_entity += " " + token_clean
                else:
                    if current_entity and current_label:
                        entities.append({
                            "label": current_label,
                            "text": current_entity.strip(),
                            "start": start_idx,
                            "end": prev_token_end
                        })
                    current_entity = token_clean
                    current_label = tag_type
                    start_idx = token_start

            prev_token_end = token_end
            idx_in_text = token_end

        if current_entity and current_label:
            entities.append({
                "label": current_label,
                "text": current_entity.strip(),
                "start": start_idx,
                "end": prev_token_end
            })

        return entities

    def extract(self, sentence):
        tokenized_input = self.tokenizer.encode(sentence)
        tags_direct = self.predict_sentence(self.model_direct, sentence, self.enc_tag_direct, self.chars_direct)
        tags_indirect = self.predict_sentence(self.model_indirect, sentence, self.enc_tag_indirect, self.chars_indirect)

        final_tags = [tag_i if tag_i in self.prefer_indirect_labels else tag_d for tag_d, tag_i in zip(tags_direct, tags_indirect)]
        token_tag_pairs = self.reverse_tokenize(tokenized_input, final_tags)
        return self.convert_to_entities(token_tag_pairs, sentence)

if __name__ == "__main__":
    extractor = QuoteEntityExtractor(
        direct_model_path="models/direct-quotes",
        indirect_model_path="models/indirect-quotes"
    )

    text = "Laporan wartawan Tribunnews.com, Fahdi Fahlevi TRIBUNNEWS.COM, JAKARTA - Ketua Umum Pengurus Besar Persatuan Guru Republik Indonesia (PGRI), Unifah Rosyidi, mengaku setuju atas rencana Pemerintah menerapkan Ujian Nasional (UN). Unifah menilai penerapan UN adalah langkah yang baik sebagai standar penilaian bagi siswa. Meski begitu, Unifah menilai UN bisa diterapkan kembali, tapi tidak menjadi satu-satunya penentu kelulusan. \"Jadi formatnya biar kan para ahli. Tapi itu diperbaiki kaya UN kayak kemarin. Enggak menjadi satu-satunya untuk lulusan. Tetapi menjadi salah satu. Bagaimanapun negara harus hadir dong. Ada standar. Kalau enggak ada standar enggak ada motivasi,\" ujar Unifah kepada wartawan, Senin (2/12/2024). Dirinya mengatakan penerapan kembali adalah upaya memperbaiki sumber daya manusia (SDM) di Indonesia. Menurut Unifah, saat ini terjadi hal yang memalukan saat pelajar Indonesia tidak bisa diterima di tingkat internasional. \"Kan malu kalau sekarang mereka tidak bisa diterima di luar negeri karena kita tidak punya dasar. Kan begitu kan. Jadi bagi kami sih yang utama adalah bagaimana dampaknya bagi masa depan bangsa. Itu yang akan kami bela,\" ucapnya. Namun, Unifah berharap penerapan UN tidak dilakukan kepada siswa Sekolah Dasar (SD). Penerapan UN, menurut Unifah, sebaiknya diterapkan pada jenjang Sekolah Menengah Pertama (SMP) dan Sekolah Menengah Atas (SMA). \"SD itu wajib belajar. Jadi mulailah di SMP. SMP kan untuk ke SMA. SMA untuk ke perguruan tinggi. Jadi seperti itu,\" kata Unifah. Pelaksanaan UN, kata Unifah, bisa dilaksanakan oleh pihak independen. Dirinya menyerahkan pelaksanaan UN dengan formulasi baru kepada Pemerintah. Para siswa, menurut Unifah, akan semangat belajar ketika UN kembali diterapkan. \" Kalau misalnya nilai UN minimum sekian untuk diterima di sini. Itu kan jadi semangat belajar. Begitu juga untuk diintegrasikan dengan perguruan tinggi,\" pungkasnya."

    segmenter = SentenceSegmentation()
    sentences = segmenter.get_sentences(text)

    for sentence in sentences:
        entities = extractor.extract(sentence)
        print(entities)