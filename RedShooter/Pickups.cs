using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Pickups : MonoBehaviour
{
    public Text scoreText;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (collision.tag == "Gem")
        {
            Scoring.totalScore++;
            scoreText.text = "SCORE: " + Scoring.totalScore;
            Debug.Log(Scoring.totalScore);
            Destroy(collision.gameObject);
        }
    }
}
