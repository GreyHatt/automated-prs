function addNumbers(a, b) {
    let sum = a + b;
    console.log('Sum is: ' sum); // Missing '+' for concatenation
  }
  
  let result = addNumbers(5, 7);
  
  if (result == 12) {
    console.log('The sum is correct');  // Should use '===' for comparison
  }
  
  let obj = {
    name: 'John',
    age: 30,
    address: '123 Main Street'
    city: 'New York' // Missing comma
  };
  
  console.log(obj['name']) // Missing semicolon at the end
  