import "./App.css";
import React from 'react';
import Container from "@mui/material/Container";
import Form from './Form';


function App({graphData}) {
    return (
        <Container>
            <Form graphData={graphData}/>
        </Container>
    );
}

export default App;