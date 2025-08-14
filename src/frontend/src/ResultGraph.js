import './App.css';
import {useLocation} from "react-router-dom";
import Container from '@mui/material/Container';
import Graph from "./Graph";

export default function ResultGraph() {
    const { state } = useLocation();

  return (
      <Container>

          <Graph triples={[state]}/>

      </Container>
      );
}