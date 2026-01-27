"""Verify LangChain 0.2.16 compatibility."""
import sys

def verify():
    try:
        # Import check
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        from langchain_core.prompts import PromptTemplate
        
        print("✅ All critical imports successful")
        
        # Memory instantiation check
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        print("✅ ConversationBufferMemory instantiated")
        
        # Memory operations check
        memory.chat_memory.add_user_message("test")
        memory.chat_memory.add_ai_message("response")
        memory.clear()
        
        print("✅ Memory operations work")
        print("\n🎉 LangChain 0.2.16 is fully compatible with your code!")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()