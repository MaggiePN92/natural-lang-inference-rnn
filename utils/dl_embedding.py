import requests, zipfile, io


def download_embedding(
    url, 
    extract_too = "/fp/projects01/ec30/magvic_large_files/modelbins"
):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(extract_too)
    print("Succesfully donwloaded word embedding.")


if __name__ == "__main__":
    # "http://vectors.nlpl.eu/repository/20/40.zip"
    # lemmatized: "http://vectors.nlpl.eu/repository/20/21.zip"
    download_embedding("http://vectors.nlpl.eu/repository/20/21.zip")
