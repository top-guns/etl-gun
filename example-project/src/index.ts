import * as rx from "rxjs";
import * as etl from 'etl-gun';


// ---
// 1. Describe all needed types

// Object type variant for the file records
type Book = {
    category: string,
    author: string,
    title: string,
    price: number
}

// Create header difinition for csv file
const header = new etl.Header(['category', 'author', 'title', { 'price': 'number' }]);


// ---
// 2. Create all needed endpoints and collections

// Create endpoint for ./data folder
const csv = new etl.Csv.Endpoint('./data');
// Create collection for the books.csv file
const books = csv.getFile('books.csv');


// ---
// 3. Create pipline to print all records from books.csv, which have 'fiction' category

// Create pipline to read all records from books.csv with skip the first record which is header and all empty lines if and
const PrintFictionsAsObjects$ = books.select(true, true).pipe(
    // Convert array type to the object type
    rx.map(v => header.arrToObj(v)),
    // Filter only records with 'fiction' category
    etl.where({category: 'fiction'}),
    // Prints founded records to the stdout
    etl.log()
);


// ---
// 3. Start pipline and wait until it has finished

await etl.run(PrintFictionsAsObjects$);
