import './App.css';
import React from "react";
import Container from '@mui/material/Container';

const ResultsList = (props) => {
  console.log(props.searchResults);

  if (!props.searchResults || !Array.isArray(props.searchResults) || props.searchResults.length === 0) {
    return <Container><span>No results found.</span></Container>;
  }

  const result = props.searchResults[0];

  if (!result || !result.Topic) {
    return <Container><span>Invalid result format.</span></Container>;
  }

  const formatDocumentId = (url) => {
    const parts = url.split('/');
    if (parts[0] === 'document') {
      const [id, version] = parts[1].split('v');
      return `arXiv:${id}${version ? ` (v${version})` : ''}`;
    }
    return url;
  };

  const getArXivUrl = (documentId) => {
    const [id] = documentId.split('v');
    return `https://arxiv.org/abs/${id}`;
  };

  return (
    <Container>
      <h5>(click on document IDs to view full content if available)</h5>
      <table className="table table-hover">
        <thead>
          <tr>
            <th>ID</th>
            <th>Document ID</th>
            <th>Content Preview</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(result.Topic).map(([id, topic], index) => {
            const topicContent = (topic && typeof topic === 'object' && topic.content)
              ? topic.content
              : (typeof topic === 'string' ? topic : JSON.stringify(topic));
            const documentId = result['URL/PDF'][index] || '';
            const formattedDocumentId = formatDocumentId(documentId);
            const url = documentId.startsWith('document/') ? getArXivUrl(documentId.split('/')[1]) : documentId;

            return (
              <tr key={id}>
                <td>{parseInt(id) + 1}</td>
                <td>
                  <a href={url} target="_blank" rel="noreferrer">
                    {formattedDocumentId}
                  </a>
                </td>
                <td>{typeof topicContent === 'string' ? topicContent.substring(0, 150) + '...' : JSON.stringify(topicContent)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Container>
  );
};

export default ResultsList;