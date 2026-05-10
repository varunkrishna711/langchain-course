
import ollama
import re
import inspect
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"



@traceable(run_type='tool')
def get_product_price(product: str) -> float:
    """Look up price of a product in catalog"""

    print(f">>> Executing get_product_price tool with argument: {product}")
    # Simulate looking up the price in a catalog
    prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199",
    }
    return float(prices.get(product, 0).strip("$"))


@traceable(run_type='tool') 
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the discounted price.
    Available tiers: Bronze, Silver, Gold"""

    print(f">>> Executing apply_discount tool with arguments: price={price}, discount={discount_tier}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price":get_product_price,
    "apply_discount":apply_discount
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)  # Gets the snippet of code without decorator
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ",".join(tools.keys())

react_prompt = f"""
                "STRICT RULES TO FOLLOW - you must follow these exactly"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price() to get the real price \n"
                "2. Only call apply_discount() after you have received"
                "a price from  get_product_price(). Pass the exact price"
                "returned by get_product_price() - do not pass a made-up number. \n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount() tool. \n"
                "4. If the use does not specify a discount tier,"
                "Ask them which tier to use - do not assume one"

                Answer the following questions as best you can. You have access to the following tools:

                {tool_descriptions}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {{question}}
                Thought: """"""

"""

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)
 

# Agent Loop 
@traceable(name="Ollama agent loop")
def run_agent(question: str):

    print(f"Question:{question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration: {iteration}")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role":"user", "content": full_prompt}],
            options = { "temperature": 0}
        )
        
        output = response.message.content
        print(f"LLM output: \n{output}")

        print(f"[Parsing]  Looking for final answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
       

        print(f"[Parsing]  Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match:
            print(f"[Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break


        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool Selected] {tool_name} with args: {tool_input_raw}")

        # raw_args = [x.strip() for x ]

    print(f"ERROR: Maximum iterations reached without final answer")
    return None

if __name__ == "__main__":
    print("Welcome to the LangChain Agent Loop with Tool Calling!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")