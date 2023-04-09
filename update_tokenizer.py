import os
import json
import tempfile

from transformers import AutoTokenizer


def change_and_save_pretrained(tokenizer, save_directory, old_tokens, new_tokens, special_tokens=None):
    with tempfile.TemporaryDirectory() as fp:
        tokenizer.save_pretrained(fp)

        with open(os.path.join(fp, 'tokenizer.json'), encoding='utf-8') as f:
            vocab = json.load(f)

    is_added = False
    for added_token in vocab['added_tokens']:
        for old_token, new_token in zip(old_tokens, new_tokens):
            if added_token['content'] == old_token:
                added_token['content'] = new_token
                is_added = True
                break

    if is_added:
        for old_token, new_token in zip(old_tokens, new_tokens):
            vocab['model']['vocab'][new_token] = vocab['model']['vocab'].pop(old_token)

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.init_kwargs['special_tokens_map_file'] = "special_tokens_map.json"
    tokenizer.init_kwargs['name_or_path'] = save_directory
    tokenizer.save_pretrained(save_directory)

    # overwrite tokenizer.json
    with open(os.path.join(save_directory, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f)


TOKENS = [
    "<sep>",
    "P01:",  # utter id1
    "P02:",  # utter id2
    "P03:",
    "P04:",
    "P05:",  # utter id1
    "P06:",  # utter id2
    "P07:",
    "P08:",
    "P09:",
    '#@주소#',
    '#@이모티콘#',
    '#@이름#',
    '#@URL#',
    '#@소속#',
    '#@기타#',
    '#@전번#',
    '#@계정#',
    '#@url#',
    '#@번호#',
    '#@금융#',
    '#@신원#',
    '#@장소#',
    '#@시스템#사진#',
    '#@시스템#동영상#',
    '#@시스템#기타#',
    '#@시스템#검색#',
    '#@시스템#지도#',
    '#@시스템#삭제#',
    '#@시스템#파일#',
    '#@시스템#송금#',
    '#@시스템#'
]


if __name__ == "__main__":
    unused_tokens = [f'<unused{i}>' for i in range(len(TOKENS))]
    special_tokens = {'sep_token': '<sep>'}

    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v1", cache_dir='cache')
    change_and_save_pretrained(tokenizer,
                               save_directory="kobart-dialogue",
                               old_tokens=unused_tokens,
                               new_tokens=TOKENS,
                               special_tokens=special_tokens)

    dialogue = [
        "<sep>P04: 하이 P01: 저기 소품샵이야<sep>P02: 어떤거<sep>P01: 3만원사야 그려주니까<sep>P02: 파는 소품샵이에여?"
        "P01: 귀여운거 엄쳥많을걸 내일 가보기로했는뎅 미리 보구올게<sep>P02: 좋짘ㅋㅋ 그림은 둘이같이 그려주는거 같은데 그쳐.",
        "P01: 대구는 비오는거 어때? 괜차나??<sep>P02: 대구는 지금 비하나두 안와여! 거긴비와여?<sep>P01: ㅇㅇ 여기 엄청와 쏟아짐<sep>P02: 대박.. 일주일쯤은 올듯ㅜ",
    ]

    print(dialogue[0])
    print('Before')
    print(tokenizer.tokenize(dialogue[0]))

    tokenizer = AutoTokenizer.from_pretrained('kobart-dialogue')
    print('After')
    print(tokenizer.tokenize(dialogue[0]))
    # print(tokenizer.special_tokens_map)
    # print(tokenizer.vocab_size)

