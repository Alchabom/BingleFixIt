<!DOCTYPE html>
<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7"> <![endif]-->
<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8"> <![endif]-->
<!--[if IE 8]>         <html class="no-js lt-ie9"> <![endif]-->
<!--[if gt IE 8]>      <html class="no-js"> <!--<![endif]-->
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title></title>
        <meta name="description" content="">
        <meta name="viewport" content="width=device-width, initial-scale=1">
         <link rel="stylesheet" href="styles.css">
    </head>

    <body>
        <!--[if lt IE 7]>
            <p class="browsehappy">You are using an <strong>outdated</strong> browser. Please <a href="#">upgrade your browser</a> to improve your experience.</p>
        <![endif]-->
        <script src="" async defer></script>
        <header class = "banner">
          
        </header>

        <div class = "description-box">
          Lorem ipsum dolor sit amet, 
          consectetur adipiscing elit. 
          Aliquam cursus lectus non tellus venenatis, vel bibendum nisi faucibus. 
          Proin vehicula dignissim magna a posuere. Donec tempus ex id arcu tempor pulvinar. 
          Phasellus mattis ullamcorper elit semper viverra. 
        </div>


        <div class = "services-section">
          <div class = "service-box">
            <img src="https://placehold.co/400x400" alt="Service 1 Badge" class="badge-icon">
            <p>Service  1 Description</p>
          </div>


        <div class = "service-box">
          <img src="https://placehold.co/400x400" alt="Service 2 Badge" class="badge-icon">
          <p>Service  2 Description</p>
        </div>



      <div class = "service-box">
        <img src="https://placehold.co/400x400" alt="Service 3 Badge" class="badge-icon">
        <p>Service  3 Description</p>
      </div>
</div> 


    </body>


    <style>
        p {
          color: black;
        }
        .paragraph1 {
          font-size: 30px;
          
        }
        .paragraph2 {
          font-size: 13px;
        }
      </style>
      <head> 
        <title>
        LeaveReview
      </title>
      </head>
      <body>
        <h1>
          Leave a Review
        </h1>
      </body>
      <head>
        <title>
          Comments
        </title>
      </head>
      <body>
        <form action = "" method = "POST">
          <label>Name:
            <input type ="text" name = "Name" required /></label><br />
          
              <label>Email:
              <input type="email" name="Email" required />
              </label><br />
          
              <label>Rating (out of 5):
              <input type="number" name="Rating" min="1" max="5" required />
              </label><br />
          
            <label>Comment:<br />
            <textarea name = "Comment" required ></textarea></label><br />
            <input type = "submit" name="Submit" value="Submit" />
          </form>

<?php
    $host = "localhost";
    $db_name = "mobile_repair";
    $username = "alchabomar@gmail.com";
    $password = "Hu9aDg-rW!LZL.W";

    $conn = new mysqli($host, $username, $password, $db_name);


    if ($conn->connect_error)
    {
      die("Connection failed: ". $conn->connect_error);
    }

    if (isset($_POST["Submit"]))
    {
      $customer_name = $_POST["Name"];
      $email = $_POST["Email"];
      $rating = $_POST["Rating"];
      $review_content = $_POST["Comment"];

      $stmt = $conn->prepare("INSERT INTO comments (customer_name, email, rating, review_content, created_at, updated_at) VALUES (?, ?, ?, ?, NOW(), NOW())");

      if ($stmt->execute())
      {
        echo "<h2>Your review has been submitted!<h2>";
      } 
      else
      {
        echo "Error: ". $stmt->error;
      }
      $stmt->close();
    }

    $sql = "SELECT customer_name, email, rating, review_content, created_at FROM comments ORDER BY created_at DESC";
    $result = $conn->query($sql);

    if ($result->num_rows > 0)
    {
      echo"<h1>Customer Reviews:</h1><hr>";

      while( $row = $result->fetch_assoc())
      {
      echo "<div class='review'>
      <span><strong>Name:</strong> " . htmlspecialchars($row["customer_name"]) . "</span><br />
      <span><strong>Email:</strong> " . htmlspecialchars($row["email"]) . "</span><br />
      <span><strong>Rating:</strong> " . htmlspecialchars($row["rating"]) . "/5</span><br />
      <span><strong>Date:</strong> " . date("Y/m/d h:i A", strtotime($row["created_at"])) . "</span><br />
      <p><strong>Review:</strong> " . htmlspecialchars($row["review_content"]) . "</p>
    </div><hr>";
      }
    }
    else
    {
      echo "No reviews yet!";
    }

    $conn->close();
?>
</body>
</html>