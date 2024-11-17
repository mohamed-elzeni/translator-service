from src.translator import translate_content
from sentence_transformers import SentenceTransformer, util
from mock import patch
import openai

model = SentenceTransformer("all-MiniLM-L6-v2")


def eval_post(post: str, expected_answer: tuple[bool, str]):
    expected_is_english, expected_translation = expected_answer

    # Generate LLM response
    llm_response = translate_content(post)
    response_is_english, response_translation = llm_response

    # Compare the boolean values
    assert expected_is_english == response_is_english

    # Encode the sentences
    expected_embedding = model.encode(expected_translation, convert_to_tensor=True)
    response_embedding = model.encode(response_translation, convert_to_tensor=True)

    # Compare cosine similarity
    similarity = util.pytorch_cos_sim(expected_embedding, response_embedding).item()
    assert similarity >= 0.6


def test_non_english1():
    eval_post("Hier ist dein erstes Beispiel.", (False, "Here is your first example."))


def test_non_english2():
    eval_post(
        "बहुत से लोग नाश्ते में अनाज खाते हैं।", (False, "Many people eat cereal for breakfast.")
    )


def test_non_english3():
    eval_post(
        "Υπάρχουν πολλά διαφορετικά είδη ζώων που ζουν στην Κίνα.",
        (False, "There are many different kinds of animals that live in China."),
    )


def test_non_english4():
    eval_post(
        "Anlamadığınız terim veya kavramlarla karşılaşırsanız anlamlarını açıklığa kavuşturmak için bir sözlüğe veya ek kaynaklara başvurun.",
        (
            False,
            "If you encounter terms or concepts you do not understand, consult a dictionary or additional resources to clarify their meanings.",
        ),
    )


def test_non_english5():
    eval_post("ตอนนี้มีหมึกสีดำอยู่ในปากกา", (False, "There is now black ink in the pen."))


def test_non_english6():
    eval_post(
        "No tengo dinero para hacer nada ahora.",
        (False, "I don't have the money to do anything right now."),
    )


def test_non_english7():
    eval_post(
        "Bomull har vackra vita och röda blommor.",
        (False, "Cotton has beautiful white and red flowers."),
    )


def test_non_english8():
    eval_post(
        "あなたはアメリカに来て貧しいかもしれませんが、一生懸命働けば、子供たちはより良い生活とより良い機会を得ることができます。",
        (
            False,
            "You may come to America and be poor, but if you work hard, your children will have a better life and better opportunities.",
        ),
    )


def test_non_english9():
    eval_post(
        "Le piramidi furono costruite sia per regine che per re, e la posizione delle regine era leggermente inferiore a quella dei loro consorti, anche se, per quanto riguarda le rappresentazioni monumentali, davano sempre la precedenza a questi ultimi.",
        (
            False,
            "The pyramids were built for both queens and kings, and the position of the queens was slightly inferior to that of their consorts, although, as regards monumental representations, they always gave precedence to the latter.",
        ),
    )


def test_non_english10():
    eval_post(
        "Bazı üniversiteler, daha yüksek ücret ödeyen uluslararası öğrencilere eğitim verdikleri yurtdışında kampüsler inşa etmiştir.",
        (
            False,
            "Some universities have built campuses abroad where they teach international students who pay higher fees.",
        ),
    )


def test_non_english11():
    eval_post(
        "El sol brilla intensamente hoy.", (False, "The sun is shining brightly today.")
    )


def test_non_english12():
    eval_post(
        "J’aime lire des livres pendant mon temps libre.",
        (False, "I like to read books in my free time."),
    )


def test_non_english13():
    eval_post(
        "Könnten Sie mir bitte den Weg zum Bahnhof zeigen?",
        (False, "Could you please show me the way to the train station?"),
    )


def test_non_english14():
    eval_post(
        "昨日、新しいレストランで美味しい寿司を食べました。",
        (False, "Yesterday, I ate delicious sushi at a new restaurant."),
    )


def test_non_english15():
    eval_post(
        "Вчера я смотрел интересный фильм.",
        (False, "Yesterday, I watched an interesting movie."),
    )


def test_english16():
    eval_post(
        "The quick brown fox jumps over the lazy dog.",
        (True, "The quick brown fox jumps over the lazy dog."),
    )


def test_english17():
    eval_post(
        "She sells seashells by the seashore.",
        (True, "She sells seashells by the seashore."),
    )


def test_english18():
    eval_post(
        "To be or not to be, that is the question.",
        (True, "To be or not to be, that is the question."),
    )


def test_english19():
    eval_post(
        "All that glitters is not gold.", (True, "All that glitters is not gold.")
    )


def test_english20():
    eval_post(
        "A journey of a thousand miles begins with a single step.",
        (True, "A journey of a thousand miles begins with a single step."),
    )


def test_english21():
    eval_post("Better late than never.", (True, "Better late than never."))


def test_english22():
    eval_post(
        "Actions speak louder than words.", (True, "Actions speak louder than words.")
    )


def test_english23():
    eval_post(
        "The pen is mightier than the sword.",
        (True, "The pen is mightier than the sword."),
    )


def test_english24():
    eval_post(
        "When in Rome, do as the Romans do.",
        (True, "When in Rome, do as the Romans do."),
    )


def test_english25():
    eval_post(
        "The early bird catches the worm.", (True, "The early bird catches the worm.")
    )


def test_english26():
    eval_post(
        "A picture is worth a thousand words.",
        (True, "A picture is worth a thousand words."),
    )


def test_english27():
    eval_post(
        "Beauty is in the eye of the beholder.",
        (True, "Beauty is in the eye of the beholder."),
    )


def test_english28():
    eval_post(
        "You can't judge a book by its cover.",
        (True, "You can't judge a book by its cover."),
    )


def test_english29():
    eval_post(
        "The grass is always greener on the other side.",
        (True, "The grass is always greener on the other side."),
    )


def test_english30():
    eval_post(
        "Don't count your chickens before they hatch.",
        (True, "Don't count your chickens before they hatch."),
    )


def test_gibberish31():
    eval_post("asdkjfhaskjdfh", (False, "LLM error: cannot translate content."))


def test_gibberish32():
    eval_post("123abc!@#", (False, "LLM error: cannot translate content."))


def test_gibberish33():
    eval_post("!@#$%^&*()", (False, "LLM error: cannot translate content."))


def test_gibberish34():
    eval_post(
        "zzzzzzzzzzeeeeeeeeeeeeeeeeeeeeeee",
        (False, "LLM error: cannot translate content."),
    )


def test_gibberish35():
    eval_post(
        "qwertyuiopasdfghjklzxcvbnm", (False, "LLM error: cannot translate content.")
    )


@patch.object(openai.ChatCompletion, "create")
def test_unexpected_language(mocker):
    mocker.return_value.choices[0].message.content = "I don't understand your request"

    assert translate_content("Hier ist dein erstes Beispiel.") == (
        False,
        "LLM error: cannot translate content.",
    )


@patch.object(openai.ChatCompletion, "create")
def test_gibberish(mocker):
    mocker.return_value.choices[0].message.content = "qwrewqerwqerwerqwerqwer"

    assert translate_content("Hier ist dein erstes Beispiel.") == (
        False,
        "LLM error: cannot translate content.",
    )


@patch.object(openai.ChatCompletion, "create")
def test_expired(mocker):
    mocker.return_value.choices[0].message.content = "The chat has expired"

    assert translate_content("Hier ist dein erstes Beispiel.") == (
        False,
        "LLM error: cannot translate content.",
    )


@patch.object(openai.ChatCompletion, "create")
def test_unable_to_handle(mocker):
    mocker.return_value.choices[0].message.content = (
        "I'm unable to handle this request at the moment"
    )

    assert translate_content("Hier ist dein erstes Beispiel.") == (
        False,
        "LLM error: cannot translate content.",
    )
