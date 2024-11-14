# Chat-Retrieval-Memory

##### This is the project USING Langchain, Qdrant and RAG 
##### Use Python Version 3.11.5 or later

### Step 1:

#### Clone this repo in VS Code or related IDE.
#### Create Virtual Env and Install these dependencies:

  1. `1pip install langchain-community`
  2. `pip install langchain`
  3. `pip install python-dotenv`
  4. `pip install langchain-openai`
  5. `pip install unstructured`
  6. `pip install "unstructured[docx]" "unstructured[pdf]" docx2txt`
  7. `pip install langchain-qdrant`
  8. `pip install qdrant-client`
  9. `pip install pypdf`

### Step2:
### --------qdrant database configuration--------

##### Download docker (i used windows, you can do respectively for Mac or Linux)

#### Open Dokcer
#### Open VS Code
##### Pull Image using command: docker pull qdrant/qdrant

##### Run the Docker Container with the Correct Path, if facing error replace below path with yours hard drive, e.g mine is 'D', your could be other then change it accordingly.

##### Open your terminal (PowerShell) and run the following command, making sure to update the path:

#### docker run -p 6333:6333 `    -v "D:\Sappyb-Chat-Retrieval-Memory\App\Sappy_data:/qdrant/storage"`
qdrant/qdrant

### To Verify Qdrant is Running

##### First, ensure that your Qdrant instance is running and accessible on port 6333. You can verify this by running the following command in your terminal:

#### curl http://localhost:6333

## Step3: 
##### Once done all above steps, now run main.py and and test it according to your data.

##### if you want to load new data, then make sure first update retreival with that data by passing its path in main.py.

##### If, confusion, then am available for meeting on zoom.

### Thanks
