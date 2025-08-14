import './App.css';
import React from "react";
import { useLocation} from "react-router-dom";
import Container from '@mui/material/Container';
import ResultsList from "./ResultsList";


export default function Results() {
    const { state } = useLocation();

    return (

        <Container>
            <h3>See your solution design recommendations </h3>
            <ResultsList searchResults={[state]}> </ResultsList>
            <a className='btn btn-primary' href="/app">Specify Your Problem Definition</a>
        </Container>
    );
}