<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Leave a Review</title>
    <meta name="description" content="">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="styles.css">

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
        /* Added this style for the reviews output by your PHP */
        .review {
            border: 1px solid #ccc;
            background: #f9f9f9;
            padding: 15px;
            margin: 15px 0;
        }
        .review p {
            margin-top: 5px;
        }
    </style>
</head>

<body>
    <script src="" async defer></script>
    
    <header class="banner">
        </header>

    <div class="description-box">
        Lorem ipsum dolor sit amet,
        consectetur adipiscing elit.
        Aliquam cursus lectus non tellus venenatis, vel bibendum nisi faucibus.
        Proin vehicula dignissim magna a posuere. Donec tempus ex id arcu tempor pulvinar.
        Phasellus mattis ullamcorper elit semper viverra.
    </div>


    <div class="services-section">
        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="Service 1 Badge" class="badge-icon">
            <p>Service 1 Description</p>
        </div>

        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="Service 2 Badge" class="badge-icon">
            <p>Service 2 Description</p>
        </div>

        <div class="service-box">
            <img src="https://placehold.co/400x400" alt="Service 3 Badge" class="badge-icon">
            <p>Service 3 Description</p>
        </div>
    </div>

    <hr> <h1>Leave a Review</h1>

    <form action="" method="POST">
        <label>Name:
            <input type="text" name="Name" required /></label><br />

        <label>Email:
            <input type="email" name="Email" required />
        </label><br />

        <label>Rating (out of 5):
            <input type="number" name="Rating" min="1" max="5" required />
        </label><br />

        <label>Comment:<br />
            <textarea name="Comment" required></textarea></label><br />
        <input type="submit" name="Submit" value="Submit" />
    </form>

    <hr> <?php
    $host = "localhost:3307";
    $db_name = "mobile_repair";
    $username = "root";
    $password = "";

    // Create connection
    $conn = new mysqli($host, $username, $password, $db_name);

    // Check connection
    if ($conn->connect_error)
    {
      die("Connection failed: ". $conn->connect_error);
    }

    // --- 1. HANDLE FORM SUBMISSION ---
    if (isset($_POST["Submit"]))
    {
      $customer_name = $_POST["Name"];
      $email = $_POST["Email"];
      $rating = $_POST["Rating"];
      $review_content = $_POST["Comment"];

      $stmt = $conn->prepare("INSERT INTO comments (customer_name, email, rating, review_content, created_at, updated_at) VALUES (?, ?, ?, ?, NOW(), NOW())");

      
      $stmt->bind_param("ssis", $customer_name, $email, $rating, $review_content);

      if ($stmt->execute())
      {
        echo "<h2>Your review has been submitted!</h2>";
      } 
      else
      {
        echo "Error: ". $stmt->error;
      }
      $stmt->close();
    }

    // --- 2. DISPLAY EXISTING REVIEWS ---
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